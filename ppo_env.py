"""Gymnasium environment for LUMEN's first PPO solar-escape experiment.

Uses the *existing* solar_sail.py and (optionally) dynamics.py on main.
No inward/outbound switching law or flyby rewards are prescribed.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.integrate import solve_ivp

from solar_sail import AU, solar_sail_accel
from dynamics import dynamics
from ephemerides import PLANET_IDS, earth_state, gm_si, load_kernels

DAY = 86400.0
YEAR = 365.25 * DAY
V_REF = 30000.0  # m/s
E_REF = 0.5 * V_REF**2  # m^2/s^2


def make_initial_state(epoch="2035-01-01T00:00:00"):
    """Exactly reproduce main.py's initial state and spacecraft parameters.

    This is a heliocentric starting condition, NOT an Earth-escape simulation.
    """
    import spiceypy as spice

    load_kernels()
    et0 = spice.str2et(epoch)
    earth0 = earth_state(et0)
    er = earth0[:3] / np.linalg.norm(earth0[:3])
    r0 = earth0[:3] + 0.02 * AU * er
    v0 = earth0[3:].copy()
    x0 = np.hstack((r0, v0))

    params = {
        "et0": et0,
        "mu_sun": gm_si(10),
        "mu_planets": np.array([gm_si(i) for i in PLANET_IDS]),
        "sail_params": (340.0, 5.0, 0.9),
    }
    return x0, params


class SolarSailPPOEnv(gym.Env):
    """One RL step = one held sail command plus numerical propagation.

    Training defaults to Sun gravity + sail for speed; the original full
    solar/planet/sail acceleration is available via include_planets=True.

    Success: outward at >= goal_au with positive solar two-body energy.
    This is an *escape milestone*, not a validated ultimate escape solution.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        x0,
        params,
        include_planets=False,
        dt_days=2.0,
        horizon_years=10.0,
        goal_au=5.0,
        min_au=0.2,
        terminal_shaping=True,
        accurate=False,
        record=False,
    ):
        super().__init__()
        self.x0 = np.asarray(x0, dtype=float).copy()
        self.params = dict(params)
        self.include_planets = bool(include_planets)
        self.dt = float(dt_days) * DAY
        self.T = float(horizon_years) * YEAR
        self.r_goal = float(goal_au) * AU
        self.r_limit = float(min_au) * AU
        self.terminal_shaping = bool(terminal_shaping)
        self.record = bool(record)
        self.mu_sun = float(params["mu_sun"])
        self.mu_planets = np.asarray(params["mu_planets"], dtype=float)
        self.sail_params = tuple(params["sail_params"])
        if self.dt <= 0 or self.T <= 0 or self.r_limit <= 0 or self.r_goal <= np.linalg.norm(self.x0[:3]):
            raise ValueError("Invalid PPO propagation duration or radial bounds")

        # Two independent continuous commands; the policy is unconstrained by
        # a manually defined inbound/outbound phase.
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        # Full J2000 state, orbital features and epoch-relative time.
        self.observation_space = spaces.Box(-20.0, 20.0, shape=(11,), dtype=np.float32)
        self.rtol = 1e-10 if accurate else 2e-7
        self.atol = ([0.1] * 3 + [1e-6] * 3) if accurate else ([20.0] * 3 + [2e-4] * 3)
        self.reset()

    def _energy(self, x):
        r = np.linalg.norm(x[:3])
        return float(0.5 * np.dot(x[3:], x[3:]) - self.mu_sun / r)

    def _observe(self):
        r, v = self.x[:3], self.x[3:]
        radius = np.linalg.norm(r)
        radial_speed = np.dot(r, v) / radius
        transverse_speed = np.linalg.norm(v - radial_speed * r / radius)
        obs = np.r_[r / AU, v / V_REF, radius / AU,
                    radial_speed / V_REF, transverse_speed / V_REF,
                    self._energy(self.x) / E_REF, self.t / self.T]
        return np.clip(obs, -20.0, 20.0).astype(np.float32)

    def _info(self, success=False):
        radius = np.linalg.norm(self.x[:3])
        energy = self._energy(self.x)
        return {
            "is_success": bool(success),
            "day": float(self.t / DAY),
            "r_au": float(radius / AU),
            "perihelion_au": float(self.r_min / AU),
            "energy_m2s2": energy,
            "v_inf_km_s": float(np.sqrt(max(0.0, 2.0 * energy)) / 1000.0),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0.0
        self.x = self.x0.copy()
        self.r_min = float(np.linalg.norm(self.x[:3]))
        self.e_initial = self._energy(self.x0)
        self.history_t = [0.0] if self.record else []
        self.history_x = [self.x.copy()] if self.record else []
        self.history_u = []
        return self._observe(), self._info()

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        if a.shape != (2,) or not np.all(np.isfinite(a)):
            raise ValueError("PPO action must contain two finite values")
        alpha = float(0.25 * np.pi * (a[0] + 1.0))  # [0, pi/2]
        delta = float(np.pi * a[1])                 # [-pi, pi]
        t_start = self.t
        t_end = min(t_start + self.dt, self.T)

        def rhs(t, x):
            if self.include_planets:
                return dynamics(t, x, self.params["et0"], self.mu_sun,
                                self.mu_planets, self.sail_params,
                                lambda _t, _x: (alpha, delta))
            r, v = x[:3], x[3:]
            a_sun = -self.mu_sun * r / np.linalg.norm(r)**3
            a_sail = solar_sail_accel(r, v, alpha, delta, *self.sail_params)
            return np.r_[v, a_sun + a_sail]

        def solar_limit(t, x):
            return np.linalg.norm(x[:3]) - self.r_limit

        solar_limit.terminal = True
        solar_limit.direction = -1

        def perihelion(t, x):
            return float(np.dot(x[:3], x[3:]) / np.linalg.norm(x[:3]))

        perihelion.terminal = False
        perihelion.direction = 1

        sol = solve_ivp(rhs, (t_start, t_end), self.x, method="DOP853",
                        events=(solar_limit, perihelion), rtol=self.rtol,
                        atol=self.atol, max_step=DAY)
        if not sol.success:
            raise RuntimeError(sol.message)

        self.t = float(sol.t[-1])
        self.x = sol.y[:, -1].copy()
        self.r_min = min(self.r_min, float(np.min(np.linalg.norm(sol.y[:3], axis=0))))
        if len(sol.y_events[1]):
            self.r_min = min(self.r_min, float(np.min(np.linalg.norm(sol.y_events[1][:, :3], axis=1))))

        energy = self._energy(self.x)
        radius = np.linalg.norm(self.x[:3])
        outward = np.dot(self.x[:3], self.x[3:]) > 0
        success = bool(radius >= self.r_goal and energy > 0 and outward)
        solar_violation = bool(len(sol.t_events[0]))
        terminated = success or solar_violation
        truncated = bool(self.t >= self.T - 1e-6 and not terminated)

        # Main objective: arrive quickly; no reward for spiraling inward.

        # Time penalty
        reward = -(self.t - t_start) / self.T

        # Net orbital energy improvement
        energy_gain = energy - self.e_initial

        # Heliocentric hyperbolic excess speed
        v_inf = np.sqrt(max(0.0, 2.0 * energy))

        if success:

            # Reward reaching 5 AU as quickly as possible
            reward += 20.0 + 10.0 * (1.0 - self.t / self.T)

            # Additional reward for a high-energy escape
            reward += 8.0 * np.tanh(v_inf / 20000.0)

        elif solar_violation:

            reward -= 20.0

        elif truncated and self.terminal_shaping:

            # Reward net energy gains even if 5 AU is not reached
            reward += 6.0 * np.tanh(energy_gain / E_REF)

        if self.record:
            self.history_t.append(self.t)
            self.history_x.append(self.x.copy())
            self.history_u.append([t_start, alpha, delta])

        return self._observe(), float(reward), terminated, truncated, self._info(success)
