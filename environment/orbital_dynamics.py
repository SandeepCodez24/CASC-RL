"""
Orbital Dynamics — Layer 1 Physical Simulation.

Implements Keplerian orbit propagation (two-body problem) with optional
J2 perturbation for higher-fidelity LEO satellite position/velocity computation.

References:
    - Vallado, D.A., "Fundamentals of Astrodynamics and Applications"
    - Curtis, H.D., "Orbital Mechanics for Engineering Students"
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


# Constants
MU_EARTH = 3.986004418e14    # m^3/s^2 — Earth gravitational parameter
R_EARTH  = 6.3781e6          # m — Earth mean radius
J2       = 1.08262668e-3     # J2 zonal harmonic coefficient


@dataclass
class OrbitalElements:
    """Classical Keplerian orbital elements."""
    semi_major_axis: float   # km
    eccentricity: float      # dimensionless [0, 1)
    inclination: float       # radians
    raan: float              # Right Ascension of Ascending Node (radians)
    arg_perigee: float       # Argument of Perigee (radians)
    true_anomaly_0: float    # Initial true anomaly (radians)


class OrbitalDynamics:
    """
    Keplerian orbit propagator for a single satellite.

    Computes position (ECI frame) and velocity at any time t given
    the satellite's classical orbital elements. Optionally applies
    J2 perturbation for secular drift in RAAN and argument of perigee.

    Args:
        semi_major_axis (float): Semi-major axis in km.
        eccentricity (float): Orbital eccentricity [0, 1).
        inclination (float): Inclination in degrees.
        raan (float): Right Ascension of Ascending Node in degrees.
        arg_perigee (float): Argument of perigee in degrees.
        true_anomaly_0 (float): Initial true anomaly in degrees.
        use_j2 (bool): Enable J2 perturbation (default False).
    """

    def __init__(
        self,
        semi_major_axis: float = 6778.0,
        eccentricity: float = 0.0,
        inclination: float = 51.6,
        raan: float = 0.0,
        arg_perigee: float = 0.0,
        true_anomaly_0: float = 0.0,
        use_j2: bool = False,
    ):
        self.a = semi_major_axis * 1e3   # Convert km → m
        self.e = eccentricity
        self.i = np.radians(inclination)
        self.raan0 = np.radians(raan)
        self.omega0 = np.radians(arg_perigee)
        self.nu0 = np.radians(true_anomaly_0)
        self.use_j2 = use_j2

        # Pre-compute orbital period
        self.period = 2 * np.pi * np.sqrt(self.a**3 / MU_EARTH)  # seconds

        # Mean motion
        self.n = 2 * np.pi / self.period  # rad/s

        # Mean anomaly at t=0 (from true anomaly)
        self.M0 = self._true_to_mean_anomaly(self.nu0)

    def _true_to_mean_anomaly(self, nu: float) -> float:
        """Convert true anomaly to mean anomaly."""
        e = self.e
        E = 2 * np.arctan2(
            np.sqrt(1 - e) * np.sin(nu / 2),
            np.sqrt(1 + e) * np.cos(nu / 2)
        )
        M = E - e * np.sin(E)
        return M % (2 * np.pi)

    def _solve_kepler(self, M: float, tol: float = 1e-8) -> float:
        """Solve Kepler's equation M = E - e*sin(E) for eccentric anomaly E."""
        E = M.copy() if hasattr(M, 'copy') else float(M)
        for _ in range(50):
            dE = (M - E + self.e * np.sin(E)) / (1 - self.e * np.cos(E))
            E += dE
            if np.all(np.abs(dE) < tol):
                break
        return E

    def _j2_rates(self) -> Tuple[float, float]:
        """Compute J2 secular drift rates for RAAN and argument of perigee."""
        p = self.a * (1 - self.e**2)
        n = self.n
        ci = np.cos(self.i)
        raan_dot = -1.5 * n * J2 * (R_EARTH / p)**2 * ci
        omega_dot = 0.75 * n * J2 * (R_EARTH / p)**2 * (5 * ci**2 - 1)
        return raan_dot, omega_dot

    def propagate(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propagate orbit to time t and return ECI position and velocity.

        Args:
            t (float): Elapsed time from epoch in seconds.

        Returns:
            position (np.ndarray): ECI position vector [x, y, z] in meters.
            velocity (np.ndarray): ECI velocity vector [vx, vy, vz] in m/s.
        """
        # Mean anomaly at time t
        M = (self.M0 + self.n * t) % (2 * np.pi)

        # Solve Kepler's equation
        E = self._solve_kepler(M)

        # True anomaly
        nu = 2 * np.arctan2(
            np.sqrt(1 + self.e) * np.sin(E / 2),
            np.sqrt(1 - self.e) * np.cos(E / 2)
        )

        # Radius
        r = self.a * (1 - self.e * np.cos(E))

        # Position in perifocal frame
        p = self.a * (1 - self.e**2)
        r_pf = np.array([r * np.cos(nu), r * np.sin(nu), 0.0])
        v_pf = np.sqrt(MU_EARTH / p) * np.array(
            [-np.sin(nu), self.e + np.cos(nu), 0.0]
        )

        # Current RAAN and omega (with optional J2)
        raan = self.raan0
        omega = self.omega0
        if self.use_j2:
            raan_dot, omega_dot = self._j2_rates()
            raan += raan_dot * t
            omega += omega_dot * t

        # Rotation matrix: perifocal → ECI
        R = self._rotation_matrix(raan, omega)

        position = R @ r_pf         # meters (ECI)
        velocity = R @ v_pf         # m/s (ECI)

        return position, velocity

    def _rotation_matrix(self, raan: float, omega: float) -> np.ndarray:
        """Construct perifocal-to-ECI rotation matrix from Euler angles."""
        ci, si = np.cos(self.i), np.sin(self.i)
        cr, sr = np.cos(raan), np.sin(raan)
        cw, sw = np.cos(omega), np.sin(omega)

        R = np.array([
            [cr*cw - sr*sw*ci,  -cr*sw - sr*cw*ci,  sr*si],
            [sr*cw + cr*sw*ci,  -sr*sw + cr*cw*ci, -cr*si],
            [sw*si,              cw*si,              ci   ],
        ])
        return R

    def get_orbital_phase(self, t: float) -> float:
        """
        Returns orbital phase as fraction of orbit [0, 2π].

        Args:
            t (float): Elapsed time in seconds.

        Returns:
            float: Mean anomaly (orbital phase) in radians [0, 2π].
        """
        return (self.M0 + self.n * t) % (2 * np.pi)

    def position_km(self, t: float) -> np.ndarray:
        """Position in km (convenience wrapper)."""
        pos, _ = self.propagate(t)
        return pos / 1e3
