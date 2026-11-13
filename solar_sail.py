import numpy as np

AU = 149597870700.0  # m
P_SRP = 4.56e-6       # N/m^2 at 1 AU


def solar_sail_accel(r, v, alpha, delta, A, m, rho):
    """Solar-sail acceleration in m/s^2, in heliocentric J2000 axes."""
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)

    er = r / np.linalg.norm(r)
    h = np.cross(r, v)
    eh = h / np.linalg.norm(h)
    et = np.cross(eh, er)

    n = (np.cos(alpha) * er
         + np.sin(alpha) * np.cos(delta) * et
         + np.sin(alpha) * np.sin(delta) * eh)

    return (P_SRP * A / m) * (AU / np.linalg.norm(r))**2 * (
        (1 - rho) * np.cos(alpha) * er
        + 2 * rho * np.cos(alpha)**2 * n
    )
