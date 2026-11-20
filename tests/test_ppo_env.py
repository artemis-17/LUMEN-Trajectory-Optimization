"""Lightweight checks for the PPO action, observations, rewards, and events."""

import unittest

import numpy as np
from stable_baselines3.common.env_checker import check_env

from ephemerides import planet_positions
from ppo_env import (
    ARRIVAL_BONUS, AU, ENERGY_BONUS, TIME_COST, VIOLATION_COST, YEAR,
    SolarSailPPOEnv, bounded_action, launch_state, make_initial_state,
    orbital_quantities, screen_route, steering_angles,
)
from solar_sail import solar_sail_accel


class SolarSailTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x0, cls.params = make_initial_state()
        cls.mu = cls.params["mu_sun"]
        cls.v_ref = np.sqrt(cls.mu / AU)

    def test_sail_commands(self):
        r = np.array([AU, 0.0, 0.0])
        v = np.array([0.0, self.v_ref, 0.0])

        def acceleration(transverse, normal=0):
            cone, clock = steering_angles(bounded_action([transverse, normal]))
            return solar_sail_accel(r, v, cone, clock, *self.params["sail_params"])

        prograde = acceleration(0.4)
        retrograde = acceleration(-0.4)
        self.assertGreater(prograde[1], 0)
        np.testing.assert_allclose(prograde, retrograde * [1, -1, 1], atol=1e-18)
        np.testing.assert_allclose(acceleration(0)[1:], 0, atol=1e-18)
        self.assertGreater(acceleration(0)[0], 0)
        np.testing.assert_allclose(acceleration(1), 0, atol=1e-18)
        normal = acceleration(0, 0.4)
        self.assertGreater(normal[2], 0)
        np.testing.assert_allclose(normal, acceleration(0, -0.4) * [1, 1, -1], atol=1e-18)
        with self.assertRaises(ValueError):
            bounded_action([np.inf, 0])

    def test_circular_orbit_and_observation_contract(self):
        state = np.array([AU, 0, 0, 0, self.v_ref, 0])
        r, vr, vt, energy, h, q, e = orbital_quantities(state, self.mu)
        np.testing.assert_allclose(
            [r / AU, vr, vt / self.v_ref, energy / (self.v_ref**2 / 2),
             h / (AU * self.v_ref), q / AU, e],
            [1, 0, 1, -1, 1, 1, 0], atol=1e-12,
        )
        for planets, size in [(False, 19), (True, 49)]:
            env = SolarSailPPOEnv(self.x0, self.params, include_planets=planets,
                                  launch_window_years=5)
            obs, _ = env.reset()
            self.assertEqual(obs.shape, (size,))
            self.assertEqual(env.action_space.shape, (2,))
            check_env(env, warn=True)

    def test_policy_selects_launch_epoch_before_flight(self):
        env = SolarSailPPOEnv(self.x0, self.params, launch_window_years=5)
        obs, _ = env.reset()
        self.assertEqual(obs[10], 1)
        obs, reward, terminated, truncated, info = env.step([0.5, 0.9])
        epoch = self.params["et0"] + 3.75 * YEAR
        self.assertEqual(info["launch_et"], epoch)
        np.testing.assert_allclose(env.x, launch_state(epoch))
        self.assertEqual((obs[9], obs[10]), (0.75, 0.0))
        self.assertEqual((env.t, reward, terminated, truncated), (0.0, 0.0, False, False))
        env.step([0, -0.4])
        self.assertEqual(env.t, env.dt)

    def test_curriculum_switches_only_on_reset(self):
        env = SolarSailPPOEnv(self.x0, self.params, include_planets=True,
                              flybys=("Jupiter", "Saturn"), curriculum=True)
        env.reset()
        self.assertEqual(env.active_goal, 2 * AU)
        env.set_curriculum(0.25)
        self.assertEqual(env.active_route, ())
        env.reset()
        self.assertEqual(env.active_route, (4,))
        env.set_curriculum(0.6)
        env.reset()
        self.assertEqual(env.active_route, (4, 5))

    def test_outer_route_screen_accepts_and_rejects_timing(self):
        route = ("Jupiter", "Saturn")
        epoch = self.params["et0"]
        self.assertIsNone(screen_route(route, epoch, YEAR, 40 * YEAR))
        self.assertEqual(screen_route(route, epoch, 0, 0.02 * YEAR), "Jupiter")

    def test_jupiter_flyby_requires_exit_and_advances_route(self):
        epoch = self.params["et0"]
        jupiter = planet_positions(epoch)[4]
        velocity = (planet_positions(epoch + 1)[4] - planet_positions(epoch - 1)[4]) / 2
        er = jupiter / np.linalg.norm(jupiter)
        eh = np.cross(jupiter, velocity)
        eh /= np.linalg.norm(eh)
        gate = 0.2 * np.linalg.norm(jupiter) * (
            self.params["mu_planets"][4] / (3 * self.mu)
        )**(1 / 3)
        start = np.r_[jupiter + gate * (1.2 * er + 0.5 * eh), velocity - 30000 * er]
        for route in (("Jupiter",), ("Jupiter", "Saturn")):
            env = SolarSailPPOEnv(start, self.params, include_planets=True,
                                  flybys=route, goal_au=8, dt_days=3, accurate=True)
            env.reset()
            for _ in range(6):
                _, _, terminated, _, info = env.step([1, 0])
                if env.encounter_entry is not None:
                    self.assertEqual(info["flyby_count"], 0)
                if info["flyby_count"]:
                    break
            self.assertEqual(info["flyby_count"], 1)
            self.assertEqual(terminated, len(route) == 1)
            self.assertEqual(info["mission_success"], len(route) == 1)
            name, entry, exit_time = env.encounters[0]
            self.assertEqual(name, "Jupiter")
            self.assertLess(entry, exit_time)

    def test_goal_event_and_energy_reward(self):
        rewards = []
        for transverse in (8000, 22000):
            env = SolarSailPPOEnv(
                [4.999 * AU, 0, 0, 10000, transverse, 0], self.params,
            )
            env.reset()
            _, reward, terminated, truncated, info = env.step([1, 0])
            self.assertTrue(terminated and not truncated)
            self.assertTrue(info["reached_goal"])
            self.assertEqual(info["positive_energy_arrival"], transverse == 22000)
            self.assertAlmostEqual(info["r_au"], 5.0)
            self.assertLess(env.t, env.dt)
            expected = (ARRIVAL_BONUS + ENERGY_BONUS * np.tanh(env.orbit[3] / env.e_ref)
                        - TIME_COST * env.t / env.T)
            self.assertAlmostEqual(reward, expected)
            rewards.append(reward)
        self.assertGreater(rewards[1], rewards[0])

    def test_earlier_arrival_wins_at_equal_energy(self):
        outcomes = []
        for radius in (4.999 * AU, 4.995 * AU):
            transverse = 8000
            radial = np.sqrt(2 * (self.mu / radius - 80e6) - transverse**2)
            env = SolarSailPPOEnv([radius, 0, 0, radial, transverse, 0], self.params)
            env.reset()
            _, reward, terminated, _, info = env.step([1, 0])
            self.assertTrue(terminated and info["reached_goal"])
            self.assertAlmostEqual(info["energy_km2_s2"], -80.0)
            outcomes.append((info["day"], reward))
        self.assertLess(outcomes[0][0], outcomes[1][0])
        self.assertGreater(outcomes[0][1], outcomes[1][1])

    def test_thermal_event(self):
        env = SolarSailPPOEnv([0.201 * AU, 0, 0, -10000, 30000, 0], self.params)
        env.reset()
        _, reward, terminated, truncated, info = env.step([1, 0])
        self.assertTrue(terminated and not truncated)
        self.assertTrue(info["thermal_violation"])
        self.assertFalse(info["reached_goal"])
        self.assertAlmostEqual(info["min_r_au"], 0.2)
        self.assertAlmostEqual(reward, -VIOLATION_COST - TIME_COST * env.t / env.T)

    def test_venus_flyby_deflects_trajectory_and_close_pass_is_penalized(self):
        epoch = self.params["et0"]
        venus = planet_positions(epoch)[1]
        velocity = (planet_positions(epoch + 1)[1] - planet_positions(epoch - 1)[1]) / 2
        er = venus / np.linalg.norm(venus)
        eh = np.cross(venus, velocity)
        eh /= np.linalg.norm(eh)
        start = np.r_[venus + er * 30e6 + eh * 20e6, velocity - 12000 * er]
        states = []
        for planets in (False, True):
            env = SolarSailPPOEnv(start, self.params, include_planets=planets, dt_days=0.05,
                                  accurate=True, record=True)
            env.reset()
            _, _, terminated, _, info = env.step([1, 0])
            self.assertFalse(terminated)
            states.append(env.x)
        self.assertGreater(np.linalg.norm(states[1][3:] - states[0][3:]), 100)
        self.assertLess(env.planet_min_m[1], 20e6)

        collision = np.r_[venus + er * 10e6, velocity - 8000 * er]
        env = SolarSailPPOEnv(collision, self.params, include_planets=True, accurate=True)
        env.reset()
        _, reward, terminated, _, info = env.step([1, 0])
        self.assertTrue(terminated and info["planet_violation"] and not info["reached_goal"])
        self.assertAlmostEqual(np.linalg.norm(env.relative_planets[1]) / 1000, 7000)
        self.assertAlmostEqual(reward, -VIOLATION_COST - TIME_COST * env.t / env.T)

    def test_shaping_telescopes_through_timeout(self):
        env = SolarSailPPOEnv(self.x0, self.params, horizon_years=0.02, shaping_coef=2)
        env.reset()
        old_phi = 0.0
        shaping_return = 0.0
        k = 0
        while True:
            start = env.t
            _, reward, done, truncated, info = env.step([-0.4, 0])
            phi = 0.0 if done else env._potential(env.orbit[5])
            extra = reward + TIME_COST * (env.t - start) / env.T
            self.assertAlmostEqual(extra, env.gamma * phi - old_phi)
            if k == 0:
                self.assertGreater(extra, 0)
            shaping_return += env.gamma**k * extra
            old_phi = phi
            k += 1
            if done:
                break
        self.assertTrue(info["timeout"])
        self.assertFalse(truncated)
        self.assertAlmostEqual(env.t, env.T)
        self.assertAlmostEqual(shaping_return, 0)


if __name__ == "__main__":
    unittest.main()
