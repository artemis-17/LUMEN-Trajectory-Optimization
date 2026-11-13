"""Trusted guided-policy loading and trajectory export."""
import json
import pickle
from pathlib import Path
import numpy as np

POLICY_KINDS={'guided_hybrid_v1','guided_hybrid_v2'}


def load_policy(path):
    """Pickle executes code: load only artifacts you trust."""
    with Path(path).open('rb') as stream:
        artifact=pickle.load(stream)
    if artifact.get('kind') not in POLICY_KINDS:
        raise ValueError('Unsupported policy: this branch only loads guided neural controllers')
    return artifact


def save_result(folder,result,report):
    folder=Path(folder)
    np.savez_compressed(folder/'trajectory.npz',t=result['t'],X=result['X'],
        commands=result['commands'],predictions=result['predictions'],
        prediction_models=result.get('prediction_models',np.array([],dtype='U32')))
    (folder/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
