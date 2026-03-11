"""
Eclipse Model — Layer 1 Physical Simulation.

Detects eclipse entry/exit based on Earth's shadow geometry in ECI frame.
Computes both umbra (full shadow) and penumbra (partial shadow) fractions.

References:
    - Montenbruck, O. & Gill, E., "Satellite Orbits" (Chapter 3)
    - Vallado, D.A., "Fundamentals of Astrodynamics and Applications"
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


# Constants
R_EARTH = 6.3781e6      # m — Earth mean radius
R_SUN   = 6.957e8       # m — Sun radius
AU      = 1.496e11      # m — 1 Astronomical Unit


@dataclass
class EclipseState:
    """Eclipse state for a single satellite at time t."""
    in_eclipse: bool        # True if in full umbra
    in_penumbra: bool       # True if in partial shadow
    eclipse_fraction: float # 0.0 = full sun, 1.0 = full umbra
    sun_vector_eci: np.ndarray  # Unit vector from Earth to Sun in ECI


class EclipseModel:
    """
    Eclipse detection using cylindrical and conical Earth shadow models.

    Determines whether a satellite at a given ECI position is in:
    - Full sunlight (eclipse_fraction = 0.0)
    - Penumbra (partial shadow, 0 < eclipse_fraction < 1)
    - Umbra (full shadow, eclipse_fraction = 1.0)

    Args:
        model (str): Shadow model — "cylindrical" (simpler) or "conical" (accurate).
    """

    def __init__(self, model: str = "conical"):
        assert model in ("cylindrical", "conical"), \
            f"model must be 'cylindrical' or 'conical', got '{model}'"
        self.model = model

    def get_sun_vector_eci(self, t_jd: float) -> np.ndarray:
        """
        Compute Sun's unit vector in ECI frame using a low-precision solar ephemeris.

        Args:
            t_jd (float): Julian date.

        Returns:
            np.ndarray: Unit vector from Earth center to Sun in ECI frame.
        """
        # Low-precision solar position (Vallado Algorithm 29)
        T_ut1 = (t_jd - 2451545.0) / 36525.0  # Julian centuries from J2000

        lambda_sun = np.radians(
            280.46 + 36000.77 * T_ut1
            + 1.914666471 * np.sin(np.radians(357.5291 + 35999.05 * T_ut1))
            + 0.019994643 * np.sin(np.radians(2 * (357.5291 + 35999.05 * T_ut1)))
        ) % (2 * np.pi)

        eps = np.radians(23.439 - 0.0130042 * T_ut1)  # obliquity

        r_sun = np.array([
            np.cos(lambda_sun),
            np.cos(eps) * np.sin(lambda_sun),
            np.sin(eps) * np.sin(lambda_sun),
        ])
        return r_sun / np.linalg.norm(r_sun)

    def check_eclipse(
        self,
        satellite_pos_eci: np.ndarray,
        sun_vector_eci: np.ndarray,
    ) -> EclipseState:
        """
        Determine eclipse state for a satellite.

        Args:
            satellite_pos_eci (np.ndarray): Satellite ECI position in meters [x,y,z].
            sun_vector_eci (np.ndarray): Unit vector from Earth to Sun (ECI).

        Returns:
            EclipseState: Eclipse state dataclass.
        """
        if self.model == "cylindrical":
            return self._cylindrical_model(satellite_pos_eci, sun_vector_eci)
        else:
            return self._conical_model(satellite_pos_eci, sun_vector_eci)

    def _cylindrical_model(
        self, r_sat: np.ndarray, r_sun_hat: np.ndarray
    ) -> EclipseState:
        """Simple cylindrical shadow model."""
        # Project satellite position onto Earth-Sun line
        proj = np.dot(r_sat, r_sun_hat)
        # Distance from shadow axis
        dist_from_axis = np.sqrt(max(0.0, np.sum(r_sat**2) - proj**2))

        in_eclipse = (proj < 0) and (dist_from_axis < R_EARTH)
        eclipse_fraction = 1.0 if in_eclipse else 0.0

        return EclipseState(
            in_eclipse=in_eclipse,
            in_penumbra=in_eclipse,
            eclipse_fraction=eclipse_fraction,
            sun_vector_eci=r_sun_hat,
        )

    def _conical_model(
        self, r_sat: np.ndarray, r_sun_hat: np.ndarray
    ) -> EclipseState:
        """
        Conical shadow model with umbra and penumbra cones.

        Based on Montenbruck & Gill, Section 3.4.
        """
        r_sat_norm = np.linalg.norm(r_sat)
        # Angle between satellite and anti-sun direction
        anti_sun = -r_sun_hat
        cos_sat = np.dot(r_sat / r_sat_norm, anti_sun)

        # Only possible eclipse if satellite is on the night side
        if cos_sat < 0:
            return EclipseState(
                in_eclipse=False,
                in_penumbra=False,
                eclipse_fraction=0.0,
                sun_vector_eci=r_sun_hat,
            )

        # Angular radii
        theta_earth = np.arcsin(R_EARTH / r_sat_norm)
        r_sun_dist = AU  # approximate
        theta_sun   = np.arcsin(R_SUN / r_sun_dist)

        # Apparent angle between Earth and Sun as seen from satellite
        theta_sat = np.arccos(cos_sat)

        # Umbra (full eclipse): Earth fully covers Sun
        in_umbra = theta_sat < (theta_earth - theta_sun)
        # Penumbra (partial eclipse): partial overlap
        in_penumbra = not in_umbra and theta_sat < (theta_earth + theta_sun)

        if in_umbra:
            eclipse_fraction = 1.0
        elif in_penumbra:
            # Interpolate
            eclipse_fraction = 1.0 - (theta_sat - (theta_earth - theta_sun)) / (
                2 * theta_sun
            )
            eclipse_fraction = float(np.clip(eclipse_fraction, 0.0, 1.0))
        else:
            eclipse_fraction = 0.0

        return EclipseState(
            in_eclipse=in_umbra,
            in_penumbra=in_penumbra or in_umbra,
            eclipse_fraction=eclipse_fraction,
            sun_vector_eci=r_sun_hat,
        )

    @staticmethod
    def eclipse_fraction_to_flag(eclipse_fraction: float) -> int:
        """Convert eclipse fraction to binary flag (1 = eclipsed, 0 = sunlit)."""
        return 1 if eclipse_fraction > 0.5 else 0
