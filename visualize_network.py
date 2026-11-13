"""Draw the residual neural controller from a trusted saved policy."""
import argparse
from pathlib import Path
from graphviz import Digraph
from graphviz.backend import ExecutableNotFound
from guided_policy import FEATURES
from policy_io import load_policy


def draw_policy(policy_path,no_view=False):
    artifact=load_policy(policy_path)
    folder=Path(policy_path).resolve().parent
    dot=Digraph('Guided_Residual_Network',format='svg')
    dot.attr(rankdir='LR')
    for i,name in enumerate(FEATURES):
        dot.node(f'i{i}',name,shape='ellipse')
    for j,label in enumerate(('cone correction: tanh','clock correction: tanh')):
        dot.node(f'o{j}',label,shape='ellipse')
        for i in range(len(FEATURES)):
            dot.edge(f'i{i}',f'o{j}',label=f'{artifact["network"].weights[j,i]:.3g}')
    dot.node('baseline','Departure / braking / orbit pumping / targeting',shape='box')
    dot.node('commands','Bounded cone and clock commands',shape='box')
    dot.edge('baseline','commands'); dot.edge('o0','commands'); dot.edge('o1','commands')
    try:
        result=dot.render(filename=str(folder/'best_network'),view=not no_view,cleanup=True)
    except ExecutableNotFound:
        result=dot.save(filename=str(folder/'best_network.dot'))
        print('Graphviz executable unavailable; saved DOT source.')
    print(result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy',default='mission_search/policy.pkl')
    parser.add_argument('--no-view',action='store_true')
    args=parser.parse_args()
    draw_policy(args.policy,args.no_view)
