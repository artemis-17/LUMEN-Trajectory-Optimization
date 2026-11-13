import numpy as np
from ephemerides import planet_positions
from solar_sail import solar_sail_accel


def dynamics(t, x, et0, mu_sun, mu_planets, sail_params, control_fn):
    r, v = x[:3], x[3:]

    a = -mu_sun * r / np.linalg.norm(r)**3

    for mu_i, r_i in zip(mu_planets, planet_positions(et0 + t)):
        dr = r_i - r
        a += mu_i * (dr / np.linalg.norm(dr)**3
                     - r_i / np.linalg.norm(r_i)**3)

    alpha, delta = control_fn(t, x)
    A, m, rho = sail_params
    a += solar_sail_accel(r, v, alpha, delta, A, m, rho)
    return np.hstack((v, a))
