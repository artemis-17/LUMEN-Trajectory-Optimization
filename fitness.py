
import numpy as np
from solar_sail import AU

def trajectory_fitness(result, params):

    # Thermal constraint
    if result["status"] == "solar_limit":
        return -100.0

    x0 = result["x0"]
    xf = result["x_final"]

    r0 = np.linalg.norm(x0[:3])
    rf = np.linalg.norm(xf[:3])
    rmin = result["r_min"]

    v0 = np.linalg.norm(x0[3:])
    vf = np.linalg.norm(xf[3:])

    tf = result["t_final"]
    tmin = result["t_r_min"]

    T = params["T"]
    rgoal = params["r_goal"]
    mu = params["mu_sun"]

    # Inbound approach
    rtarget = 0.23 * AU

    pin = np.clip(
        (r0 - rmin) / (r0 - rtarget),
        0.0, 1.0
    )

    Fin = 5.0*pin + 2.0*pin*(1.0 - tmin/T)

    # Outbound progress
    pout = np.clip(
        (rf - rmin) / (rgoal - rmin),
        0.0, 1.0
    )

    Fout = 5.0 * pin**2 * pout

    # Specific orbital energy gain
    E0 = v0**2/2 - mu/r0
    Ef = vf**2/2 - mu/rf

    Eref = 0.5 * 30000.0**2

    Fenergy = 2.0 * pin**2 * pout * np.tanh(
        (Ef - E0) / Eref
    )

    # Successful arrival at 5 AU
    Fgoal = 0.0

    if result["status"] == "reached_goal":
        Fgoal = pin**2 * (
            20.0 + 10.0*(1.0 - tf/T)
        )

    return float(Fin + Fout + Fenergy + Fgoal)
