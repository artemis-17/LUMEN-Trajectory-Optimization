"""Persistent DOP853 with steps bounded at each held-command discontinuity."""
import numpy as np
from scipy.integrate import DOP853
from scipy.optimize import brentq
from simulation import rhs
from ephemerides import near_planet


def simulate_persistent(control_fn, x0, params, save_history=False):
    x0 = np.asarray(x0, dtype=float)
    T, interval = params['T'], params['dt_control']
    if T <= 0 or interval <= 0:
        raise ValueError('Propagation horizon and command interval must be positive')
    angles = [0.0, 0.0]
    def held_control(t, x):
        return angles
    def fun(t, x):
        return rhs(t, x, params['et0'], params['mu_sun'], params['mu_planets'],
                   params['sail_params'], held_control)
    def radius(x):
        return np.linalg.norm(x[:3])
    def radial(x):
        return np.dot(x[:3], x[3:])/radius(x)
    def command(t, x):
        a, d = map(float, control_fn(t, x))
        if not np.all(np.isfinite([a, d])) or not 0 <= a <= np.pi/2:
            raise ValueError('Invalid solar sail control angles')
        angles[:] = a, d
    command(0.0, x0)
    boundary = min(interval, T)
    solver = DOP853(fun, 0.0, x0, boundary, rtol=params['rtol'], atol=params['atol'],
                    max_step=params['max_step'])
    times, states = [0.0], [x0.copy()]
    r_min, t_r_min = radius(x0), 0.0
    x_r_min = x0.copy()
    status = 'time_limit'
    sample_step = boundary/max(1, int(np.ceil(boundary/params['plot_dt'])))
    next_sample = sample_step
    final_t, final_x = 0.0, x0.copy()
    while True:
        left_t, left_x = solver.t, solver.y.copy()
        if params.get('close_guard_au'):
            close = near_planet(params['et0']+left_t, left_x[:3], params['close_guard_au'])
            solver.rtol = params['close_rtol'] if close else params['rtol']
            solver.atol = np.asarray(params['close_atol'] if close else params['atol'])
        message = solver.step()
        if solver.status == 'failed':
            raise RuntimeError(message)
        right_t, right_x = solver.t, solver.y.copy()
        dense = None
        def interpolate(t):
            nonlocal dense
            if dense is None:
                dense = solver.dense_output()
            return dense(t)
        roots = []
        for limit, direction, label in ((params['r_min'], -1, 'solar_limit'),
                                        (params['r_goal'], 1, 'reached_goal')):
            a, b = radius(left_x)-limit, radius(right_x)-limit
            if (direction < 0 and a >= 0 and b <= 0) or (direction > 0 and a <= 0 and b >= 0):
                event_t = brentq(lambda t: radius(interpolate(t))-limit, left_t, right_t, xtol=1e-6)
                roots.append((event_t, label))
        if roots:
            right_t, status = min(roots)
            right_x = interpolate(right_t)
        # Locate perihelion within the accepted step for consistent fitness metrics.
        if radial(left_x) < 0 <= radial(right_x):
            peri_t = brentq(lambda t: radial(interpolate(t)), left_t, right_t, xtol=1e-6)
            peri_r = radius(interpolate(peri_t))
            if peri_r < r_min:
                r_min, t_r_min = peri_r, peri_t
                x_r_min = interpolate(peri_t).copy()
        if radius(right_x) < r_min:
            r_min, t_r_min = radius(right_x), right_t
            x_r_min = right_x.copy()
        if save_history:
            while next_sample <= right_t + 1e-7:
                sample_t = min(next_sample, right_t)
                if sample_t > times[-1]+1e-7:
                    times.append(sample_t); states.append(interpolate(sample_t).copy())
                next_sample += sample_step
            if roots and right_t > times[-1]+1e-7:
                times.append(right_t); states.append(right_x.copy())
        final_t, final_x = float(right_t), right_x.copy()
        if roots:
            break
        if right_t >= T-1e-7:
            break
        if solver.status == 'finished':
            # Keep the solver/adaptive step estimate, but refresh the cached
            # derivative because the held control has changed discontinuously.
            on_grid = abs(right_t/interval-round(right_t/interval)) < 1e-8
            if on_grid:
                command(right_t, right_x)
            boundary = min((np.floor(right_t/interval+1e-8)+1)*interval,T)
            solver.t_bound = boundary
            solver.status = 'running'
            solver.f = solver.fun(right_t, right_x)
            sample_step = (boundary-right_t)/max(1, int(np.ceil((boundary-right_t)/params['plot_dt'])))
            next_sample = right_t+sample_step
    if save_history and final_t > times[-1]+1e-7:
        times.append(final_t); states.append(final_x.copy())
    return dict(status=status, x0=x0, x_final=final_x, t_final=final_t,
                r_min=r_min, t_r_min=t_r_min, nfev=solver.nfev,
                x_r_min=x_r_min,
                t=np.asarray(times) if save_history else None,
                X=np.asarray(states).T if save_history else None)
