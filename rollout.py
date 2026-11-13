"""Connect neural guidance, numerical propagation, and mission fitness."""
import numpy as np
from fitness import assess,score_trajectory
from simulation import simulate_trajectory
from ephemerides import force_ephemeris

def rollout(net, mission, x0, params, phase='mission', table=None):
    control = net.make_control(mission,params)
    with force_ephemeris(table):
        result = simulate_trajectory(control, x0, params, save_history=True)
    result['commands'] = np.asarray(control.commands)
    result['predictions'] = np.asarray(control.predictions, dtype=float).reshape(-1, 4)
    result['prediction_models'] = np.asarray(control.prediction_models,dtype='U32')
    report=assess(result,mission,params)
    report['search_phase']=phase
    report['fitness']=score_trajectory(report,result,phase,mission)
    return result,report
