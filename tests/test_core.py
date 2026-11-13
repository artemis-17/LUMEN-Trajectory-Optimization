"""Essential regression checks: physics, objective, budget and saved replay.

Run: python -m unittest discover -s tests -v
"""
from pathlib import Path
from tempfile import TemporaryDirectory
import contextlib
import io
import unittest
import numpy as np
import spiceypy as spice
from control import work_maximizing_cone
from encounters import lambert_departure_velocity,solar_coast_positions
from fitness import score_trajectory,assess
from guided_policy import GuidedNetwork,FEATURES
from mission import Mission,setup,AU,DAY,YEAR
from optimizer import train,replay,decode,epoch_bounds,seed_population,DIMENSIONS
from policy_io import load_policy
from search import SearchSettings
from simulation import simulate_trajectory
from solar_sail import solar_sail_accel


class CoreTests(unittest.TestCase):
    def test_saved_settings_filter_retired_fields_only_for_artifacts(self):
        settings=SearchSettings.from_saved({'require_stage_success':True,'screen_rtol':1e-7})
        self.assertEqual(settings.screen_rtol,1e-7)
        with self.assertRaises(TypeError):
            SearchSettings(require_stage_success=True)
        self.assertFalse(hasattr(Mission.from_saved({'epoch':'2035-01-01','prediction_days':180}),'prediction_days'))

    def test_epoch_seeds_and_fixed_date(self):
        mission=Mission(years=.001,targets=[],epoch_start='2034-01-01',epoch_end='2036-01-01')
        setup(mission); bounds=epoch_bounds(mission)
        population=seed_population(6,mission,bounds,np.random.default_rng(1))
        dates=[spice.str2et(decode(v,mission,bounds)[0].epoch) for v in population]
        self.assertAlmostEqual(min(dates),bounds[0],places=3)
        self.assertAlmostEqual(max(dates),bounds[1],places=3)
        fixed=Mission(years=.001,targets=[])
        bounds=epoch_bounds(fixed)
        for v in (np.zeros(DIMENSIONS),np.ones(DIMENSIONS)):
            self.assertAlmostEqual(spice.str2et(decode(v,fixed,bounds)[0].epoch),bounds[0],places=3)

    def test_departure_work_and_network_bounds(self):
        r=np.array([AU,0,0]); v=np.array([0,30000,0])
        for radial in (-2000,0,2000):
            angle=work_maximizing_cone(radial,500,.9)
            work=np.dot(solar_sail_accel(r,v,angle,0,340,5,.9),[radial,500,0])
            self.assertGreaterEqual(work,-1e-12)
        output=GuidedNetwork(weights=np.full((2,len(FEATURES)),.6)).activate(np.ones(len(FEATURES)))
        self.assertTrue(np.all(np.abs(output)<=1))

    def test_sun_coast_and_lambert_boundary(self):
        mu=1.32712440018e20
        r=np.array([AU,0,0]); target=np.array([0,AU,0])
        tof=np.pi/2*np.sqrt(AU**3/mu)
        velocity=lambert_departure_velocity(r,target,tof,mu,np.array([0,0,1]))
        end=solar_coast_positions(np.r_[r,velocity],[tof],mu)[0]
        self.assertLess(np.linalg.norm(end-target),100)

    def test_persistent_reference_agreement_and_minimum(self):
        x0,params=setup(Mission(years=5/365.25,targets=[]))
        control=lambda t,x:(.6,1.2*(-1)**int(round(t/DAY)))
        reference=simulate_trajectory(control,x0,dict(params,backend='reference'),True)
        persistent=simulate_trajectory(control,x0,dict(params,backend='persistent'),True)
        self.assertEqual(persistent['status'],reference['status'])
        self.assertLess(np.linalg.norm(persistent['x_final'][:3]-reference['x_final'][:3]),100)
        self.assertLess(persistent['nfev'],reference['nfev'])

    def test_science_miss_is_not_bypassed_for_escape(self):
        state=np.array([AU,0,0,0,30000,0])
        result=dict(x0=state,x_final=np.array([100*AU,0,0,30000,0,0]))
        report=dict(propagation_status='reached_goal',encounters=[dict(closest_sample_au=3.5)],
                    vinf_kms=25,completed=0,feasible=False)
        bypass=score_trajectory(report,result,'mission',Mission())
        close=dict(report,propagation_status='time_limit',encounters=[dict(closest_sample_au=1.9)],vinf_kms=0)
        self.assertGreater(score_trajectory(close,result,'mission',Mission()),bypass)

    def test_budget_archive_and_exact_replay(self):
        mission=Mission(years=.001,targets=[],epoch_start='2035-01-01',epoch_end='2035-01-03')
        with TemporaryDirectory(dir=Path.cwd()) as folder,contextlib.redirect_stdout(io.StringIO()):
            settings=SearchSettings(population=4,generations=2,workers=1,max_evaluations=12,interpolate=False)
            result,report,_=train(mission,output=folder,settings=settings)
            self.assertLessEqual(report['training_rollouts'],12)
            artifact=load_policy(Path(folder,'policy.pkl'))
            self.assertEqual(artifact['network'].weights.shape,(2,8))
            repeated,rereport=replay(Path(folder,'policy.pkl'))
            np.testing.assert_array_equal(result['X'],repeated['X'])
            np.testing.assert_array_equal(result['commands'],repeated['commands'])
            self.assertEqual(report['fitness'],rereport['fitness'])
            self.assertFalse(report['feasible'])


if __name__=='__main__':
    unittest.main()
