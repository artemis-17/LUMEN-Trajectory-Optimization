"""Benchmark reference, persistent and screened propagation; report numerical error."""
import argparse
import json
from time import perf_counter
import numpy as np
from scipy.stats import spearmanr
from ephemerides import EphemerisTable, exact_planet_positions, body_state
from mission import Mission, setup, AU
from rollout import rollout
from search import SearchSettings, rollout_parameters
from guided_policy import GuidedNetwork


def benchmark(days=30, policies=4, repeats=2, output=None):
    if days <= 0 or policies < 1 or repeats < 1:
        raise ValueError('Benchmark dimensions must be positive')
    mission = Mission(years=days/365.25,targets=['jupiter'])
    x0,base = setup(mission)
    settings = SearchSettings()
    start = perf_counter()
    table = EphemerisTable(base['et0'],base['T'],settings.ephemeris_step_days)
    build_seconds = perf_counter()-start
    nets = [GuidedNetwork(switch_perihelion_au=.18+.12*i/max(1,policies-1),
                         brake_cone_deg=30+10*i/max(1,policies-1)) for i in range(policies)]
    radial = x0[:3]/np.linalg.norm(x0[:3])
    cases = {'departure':x0, 'cruise':x0+np.r_[.2*AU*radial,np.zeros(3)]}
    report = dict(days=days, distinct_policies=policies,repeats=repeats,
                  ephemeris_build_seconds=build_seconds,ephemeris_audit=table.audit,cases={})
    for case,initial in cases.items():
        timings = {mode:[] for mode in ('reference','accurate','screen')}
        results, scores = {}, {}
        for repeat in range(repeats):
            order = ('reference','accurate','screen') if repeat%2==0 else ('screen','accurate','reference')
            for mode in order:
                exact_planet_positions.cache_clear(); body_state.cache_clear()
                params = rollout_parameters(base,settings,base['T'],mode)
                started = perf_counter()
                pairs = [rollout(net,mission,initial,params,table=table if mode=='screen' else None) for net in nets]
                timings[mode].append(perf_counter()-started)
                results[mode] = [pair[0] for pair in pairs]
                scores[mode] = [pair[1]['fitness'] for pair in pairs]
        values = {mode:float(np.median(times)) for mode,times in timings.items()}
        errors = {}
        for mode in ('accurate','screen'):
            errors[mode] = dict(
                max_final_position_difference_m=float(max(np.linalg.norm(a['x_final'][:3]-b['x_final'][:3])
                    for a,b in zip(results[mode],results['reference']))),
                max_final_velocity_difference_mps=float(max(np.linalg.norm(a['x_final'][3:]-b['x_final'][3:])
                    for a,b in zip(results[mode],results['reference']))),
                termination_status_agreement=all(a['status']==b['status'] for a,b in zip(results[mode],results['reference'])),
                fitness_rank_correlation=(float(spearmanr(scores[mode],scores['reference']).statistic)
                    if policies>1 and np.std(scores[mode])>0 and np.std(scores['reference'])>0 else None))
        report['cases'][case] = dict(seconds=values,
            persistent_speedup=values['reference']/values['accurate'],
            screening_speedup=values['reference']/values['screen'],
            total_rhs_evaluations={mode:sum(r['nfev'] for r in items) for mode,items in results.items()},
            comparisons=errors)
    text = json.dumps(report,indent=2)
    print(text)
    if output:
        from pathlib import Path
        Path(output).write_text(text,encoding='utf-8')
    return report


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--days',type=float,default=30)
    parser.add_argument('--policies',type=int,default=4)
    parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--output')
    args = parser.parse_args()
    benchmark(args.days,args.policies,args.repeats,args.output)
