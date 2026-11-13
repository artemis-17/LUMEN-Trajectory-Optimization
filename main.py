"""Train or replay the guided neural trajectory optimizer."""
import argparse
import json
from pathlib import Path
from mission import Mission
from search import SearchSettings
from optimizer import train,replay


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mission',default=str(Path(__file__).with_name('mission_example.json')))
    parser.add_argument('--search',default=str(Path(__file__).with_name('search_settings.json')))
    parser.add_argument('--epoch-start')
    parser.add_argument('--epoch-end')
    parser.add_argument('--population',type=int,help='Override population in search settings')
    parser.add_argument('--generations','--rounds',dest='generations',type=int,
                        help='Override maximum search rounds in search settings')
    parser.add_argument('--workers',type=int)
    parser.add_argument('--max-evaluations',type=int,help='Training rollout cap; one final reference validation is extra')
    parser.add_argument('--seed',type=int)
    parser.add_argument('--seed-policy',help='One trusted saved guided controller to seed the search')
    parser.add_argument('--output',default='mission_search')
    parser.add_argument('--replay',help='Trusted saved guided policy.pkl')
    parser.add_argument('--full',action='store_true',help='With --replay, evaluate the full requested mission')
    parser.add_argument('--animate',action='store_true')
    args=parser.parse_args()
    if args.full and not args.replay:
        parser.error('--full is only used with --replay')
    if args.replay:
        replay(args.replay,args.animate,args.full)
        return
    mission=Mission(**json.loads(Path(args.mission).read_text(encoding='utf-8-sig')))
    for key in ('epoch_start','epoch_end'):
        if getattr(args,key) is not None:
            setattr(mission,key,getattr(args,key))
    settings=SearchSettings(**json.loads(Path(args.search).read_text(encoding='utf-8-sig')))
    for key in ('population','generations','workers','max_evaluations','seed'):
        if getattr(args,key) is not None:
            setattr(settings,key,getattr(args,key))
    settings.validate()
    result,report,_=train(mission,output=args.output,settings=settings,seed_policy=args.seed_policy)
    if args.animate:
        import spiceypy as spice
        from plotting import animate_rollout
        animate_rollout(result,spice.str2et(report['epoch']),Path(args.output)/'animation.html')


if __name__=='__main__':
    main()
