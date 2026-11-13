
import numpy as np
from scipy.integrate import solve_ivp
from dynamics import dynamics
from ephemerides import set_force_position


def rhs(t, x, *args):
    set_force_position(x[:3])
    return dynamics(t, x, *args)

def simulate_reference(control_fn, x0, params, save_history=False):

    def solar_event(t, x, *args):
        return np.linalg.norm(x[:3]) - params["r_min"]

    solar_event.terminal = True
    solar_event.direction = -1

    def goal_event(t, x, *args):
        return np.linalg.norm(x[:3]) - params["r_goal"]

    goal_event.terminal = True
    goal_event.direction = 1

    def perihelion_event(t, x, *args):
        return np.dot(x[:3], x[3:]) / np.linalg.norm(x[:3])

    perihelion_event.terminal = False
    perihelion_event.direction = 1

    t = 0.0
    x = np.asarray(x0, dtype=float).copy()

    t_hist = [t]
    x_hist = [x.copy()]

    r_min = np.linalg.norm(x[:3])
    x_r_min = x.copy()
    t_r_min = 0.0
    status = "time_limit"
    nfev = 0
    end_time = params['T']
    next_command = 0.0

    args_common = (
        params["et0"],
        params["mu_sun"],
        params["mu_planets"],
        params["sail_params"]
    )

    while t < end_time - 1e-7:

        if t >= next_command-1e-7:
            alpha, delta = map(float, control_fn(t, x))
            next_command = t+params['dt_control']

        if not np.all(np.isfinite([alpha, delta])) or not 0 <= alpha <= np.pi/2:
            raise ValueError("Invalid solar sail control angles")

        # Hold the commanded angles during this guidance interval
        def held_control(tt, xx):
            return alpha, delta

        t_next = min(next_command,end_time)

        sol = solve_ivp(
            rhs, (t, t_next), x,
            args=(*args_common, held_control),
            events=(solar_event, goal_event, perihelion_event),
            method="DOP853",
            dense_output=save_history,
            rtol=params["rtol"],
            atol=params["atol"],
            max_step=params["max_step"]
        )

        if not sol.success:
            raise RuntimeError(sol.message)
        nfev += sol.nfev

        # Track closest solar approach
        radii = np.linalg.norm(sol.y[:3], axis=0)
        i = np.argmin(radii)

        if radii[i] < r_min:
            r_min = radii[i]
            t_r_min = sol.t[i]
            x_r_min = sol.y[:,i].copy()

        for ti, xi in zip(sol.t_events[2], sol.y_events[2]):
            ri = np.linalg.norm(xi[:3])
            if ri < r_min:
                r_min = ri
                t_r_min = ti
                x_r_min = xi.copy()

        t_end = sol.t[-1]

        if save_history:
            n = max(2, int(np.ceil(
                (t_end - t) / params["plot_dt"]
            )) + 1)

            t_plot = np.linspace(t, t_end, n)[1:]
            X_plot = sol.sol(t_plot).T

            t_hist.extend(t_plot)
            x_hist.extend(X_plot)

        t = float(t_end)
        x = sol.y[:, -1].copy()

        if len(sol.t_events[0]) > 0:
            status = "solar_limit"
            break

        if len(sol.t_events[1]) > 0:
            status = "reached_goal"
            break

    return {
        "status": status,
        "nfev": nfev,
        "x0": np.asarray(x0),
        "x_final": x,
        "t_final": t,
        "r_min": r_min,
        "t_r_min": t_r_min,
        "x_r_min": x_r_min,
        "t": np.array(t_hist) if save_history else None,
        "X": np.array(x_hist).T if save_history else None
    }


def simulate_trajectory(control_fn, x0, params, save_history=False):
    if params.get('backend', 'persistent') == 'reference':
        return simulate_reference(control_fn, x0, params, save_history)
    from integrator import simulate_persistent
    return simulate_persistent(control_fn, x0, params, save_history)
