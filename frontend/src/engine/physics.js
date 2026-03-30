
// Orbital mechanics engine — mirrors orbital_dynamics.py and eclipse_model.py
// Constants match Python training environment exactly

export const MU_EARTH = 3.986004418e14; // m^3/s^2
export const R_EARTH  = 6.3781e6;       // m
export const J2       = 1.08262668e-3;  // J2 zonal harmonic

/**
 * Propagate satellite position in ECI frame via Keplerian two-body.
 * Solves Kepler's equation via Newton iterations.
 */
export function propagateOrbit(elements, t) {
  const { a, e, i, raan0, omega0, M0, n } = elements;
  const M = ((M0 + n * t) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI);

  // Newton solve for eccentric anomaly E
  let E = M;
  for (let j = 0; j < 15; j++) {
    E = E - (E - e * Math.sin(E) - M) / (1 - e * Math.cos(E));
  }

  // True anomaly
  const nu = 2 * Math.atan2(
    Math.sqrt(1 + e) * Math.sin(E / 2),
    Math.sqrt(1 - e) * Math.cos(E / 2)
  );

  const r = a * (1 - e * Math.cos(E));
  const r_pf = [r * Math.cos(nu), r * Math.sin(nu), 0];
  const R = rotationMatrix(raan0, i, omega0);

  return {
    x: R[0][0]*r_pf[0] + R[0][1]*r_pf[1],
    y: R[1][0]*r_pf[0] + R[1][1]*r_pf[1],
    z: R[2][0]*r_pf[0] + R[2][1]*r_pf[1] + R[2][2]*0,
    r,
  };
}

function rotationMatrix(raan, i, omega) {
  const cr=Math.cos(raan), sr=Math.sin(raan);
  const ci=Math.cos(i),    si=Math.sin(i);
  const cw=Math.cos(omega),sw=Math.sin(omega);
  return [
    [cr*cw - sr*sw*ci, -cr*sw - sr*cw*ci, sr*si],
    [sr*cw + cr*sw*ci, -sr*sw + cr*cw*ci, -cr*si],
    [sw*si,             cw*si,             ci   ]
  ];
}

/**
 * Cylindrical shadow model for eclipse detection.
 * Sun is simplified along +X ECI axis.
 */
export function isSatelliteInEclipse(pos) {
  if (pos.x < 0) {
    const d = Math.sqrt(pos.y ** 2 + pos.z ** 2);
    if (d < R_EARTH) return true;
  }
  return false;
}

/** Compute fraction of orbit before eclipse entry (0–1). Simplified. */
export function timeToEclipse(elements, currentT) {
  // Scan next 6000 seconds (one orbit ≈ 5640s)
  const step = 30;
  const currEclipse = isSatelliteInEclipse(propagateOrbit(elements, currentT));
  for (let dt = step; dt < 6000; dt += step) {
    const inEclipse = isSatelliteInEclipse(propagateOrbit(elements, currentT + dt));
    if (!currEclipse && inEclipse) return dt;  // seconds to eclipse entry
    if (currEclipse && !inEclipse) return -dt; // negative = seconds in eclipse
  }
  return null;
}

/** Generate orbital elements for a Walker-Delta constellation. */
export function walkerDelta(count = 12, altitude_m = 7000e3, inclination_rad = 0.9) {
  const n = Math.sqrt(MU_EARTH / (altitude_m ** 3));
  const sats = [];
  for (let i = 0; i < count; i++) {
    const planeIdx = Math.floor(i / (count / 4));
    const satInPlane = i % (count / 4);
    sats.push({
      a: altitude_m,
      e: 0.001,
      i: inclination_rad,
      raan0: (planeIdx / 4) * 2 * Math.PI,
      omega0: 0,
      M0: (satInPlane / (count / 4)) * 2 * Math.PI + (planeIdx * Math.PI / count),
      n,
    });
  }
  return sats;
}
