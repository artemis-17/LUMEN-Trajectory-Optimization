"""Budgeted joint epoch/steering search from physically informed neural policies.

Each candidate flies the full gravity/SRP model. Differential evolution selects
using screening scores; accurate checks build a separate candidate archive.
"""
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict,replace
from pathlib import Path
from time import perf_counter
import json,pickle
import hashlib,importlib.metadata
import numpy as np
import spiceypy as spice
from ephemerides import EphemerisTable,load_kernels
from guided_policy import GuidedNetwork,FEATURES
from mission import Mission,setup,AU,YEAR
from search import SearchSettings,rollout_parameters
from rollout import rollout
from policy_io import load_policy,save_result

PARAMETERS=6
DIMENSIONS=PARAMETERS+2*len(FEATURES)
_worker=None


def epoch_bounds(mission):
    if (mission.epoch_start is None)!=(mission.epoch_end is None):
        raise ValueError('Provide both epoch_start and epoch_end, or neither')
    lower=float(spice.str2et(mission.epoch_start or mission.epoch))
    upper=float(spice.str2et(mission.epoch_end or mission.epoch))
    if upper<lower:
        raise ValueError('epoch_end precedes epoch_start')
    return lower,upper


def decode(vector,mission,bounds):
    v=np.clip(np.asarray(vector,dtype=float),0,1)
    if v.shape!=(DIMENSIONS,) or not np.isfinite(v).all():
        raise ValueError('Invalid candidate vector')
    epoch=spice.et2utc(bounds[0]+v[0]*(bounds[1]-bounds[0]),'ISOC',3)
    selected=replace(mission,epoch=epoch,departure_phase_deg=float(360*v[1]))
    network=GuidedNetwork(switch_perihelion_au=float(.12+.22*v[2]),
        brake_cone_deg=float(25+20*v[3]),reset_radius_au=float(.7+2.3*v[4]),
        targeting_gain=float(3*v[5]),weights=(v[PARAMETERS:].reshape(2,len(FEATURES))-.5)*1.2)
    return selected,network


def initialize_worker(mission,settings,bounds,max_horizon):
    global _worker
    load_kernels()
    table=None
    if settings.interpolate:
        # One shared time span per process, not one table per trial departure date.
        table=EphemerisTable(bounds[0],bounds[1]-bounds[0]+max_horizon,
            settings.ephemeris_step_days,max(settings.ephemeris_guard_au,3*mission.flyby_radius_au),
            settings.ephemeris_error_limit_m)
    _worker=mission,settings,bounds,table


def evaluate(job):
    key,vector,phase,horizon,fidelity=job
    mission,settings,bounds,table=_worker
    start=perf_counter()
    try:
        selected,network=decode(vector,mission,bounds)
        x0,base=setup(selected,initialize=False)
        params=rollout_parameters(base,settings,horizon,fidelity)
        params['progress_label']=f'{phase}/{fidelity} candidate {key}, {selected.epoch[:10]}'
        result,report=rollout(network,selected,x0,params,phase=phase,table=table if fidelity=='screen' else None)
        report.update(search_phase=phase,epoch=selected.epoch,
            departure_phase_deg=selected.departure_phase_deg,elapsed_seconds=perf_counter()-start,
            search_score=report['fitness'],fidelity=fidelity)
        if not np.isfinite(report['search_score']):
            raise ValueError('Nonfinite hybrid objective')
        if fidelity=='screen':
            report['candidate_meets_criteria']=report['feasible']
            report['feasible']=False
            report['status']='screen_only'
        return key,report,None
    except (ValueError,RuntimeError,FloatingPointError) as exc:
        return key,None,str(exc)


def schedule(mission,generations):
    stages=[('scout',min(4,mission.years)*YEAR),
            ('encounter',min(12,mission.years)*YEAR),('mission',mission.years*YEAR)]
    if mission.years<=4:
        return [('mission',mission.years*YEAR)]*generations
    if generations==1:
        return [stages[0]]
    if generations==2:
        return [stages[0],stages[1]]
    first=max(1,int(round(generations*.4))); second=max(1,int(round(generations*.4)))
    if first+second>=generations:
        second=generations-first-1
    return [stages[0]]*first+[stages[1]]*second+[stages[2]]*(generations-first-second)


def seed_population(size,mission,bounds,rng):
    population=np.full((size,DIMENSIONS),.5)
    dates=np.linspace(0,1,size)
    q_values=[.18,.22,.26,.30,.16,.24]
    original=float(spice.str2et(mission.epoch))
    for i,row in enumerate(population):
        row[0]=dates[i%len(dates)]
        row[1]=(mission.departure_phase_deg/360+i*.381966)%1
        row[2]=(q_values[i%len(q_values)]-.12)/.22
        row[3]=(35.264-25)/20
        row[4]=(1.0-.7)/2.3
        row[5]=[0,0,.5,1.5,0,3][i%6]/3
        if i>=6:
            row[PARAMETERS:]=np.clip(.5+rng.normal(0,.08,DIMENSIONS-PARAMETERS),0,1)
    if bounds[1]>bounds[0] and bounds[0]<=original<=bounds[1]:
        population[0,0]=(original-bounds[0])/(bounds[1]-bounds[0])
    if size>=3:
        population[1,0]=0; population[-1,0]=1
    for i in range(1,size,2):
        population[i,PARAMETERS+len(FEATURES)+5]=.8
    return population


def train(mission,generations=None,population_size=None,workers=None,seed=None,
          output='mission_search',settings=None,max_evaluations=None,seed_policy=None):
    settings=settings or SearchSettings()
    generations=settings.generations if generations is None else generations
    population_size=settings.population if population_size is None else population_size
    workers=settings.workers if workers is None else workers
    seed=settings.seed if seed is None else seed
    max_evaluations=settings.max_evaluations if max_evaluations is None else max_evaluations
    settings=replace(settings,generations=generations,population=population_size,workers=workers,
                     seed=seed,max_evaluations=max_evaluations)
    settings.validate()
    mission.validate(); load_kernels()
    bounds=epoch_bounds(mission)
    # Fail early on unavailable end-of-window mission coverage.
    setup(replace(mission,epoch=spice.et2utc(bounds[1],'ISOC',3)),initialize=False)
    if generations<1 or population_size<1 or workers<1 or max_evaluations<population_size+1:
        raise ValueError('Positive population/generations/workers required; budget must cover population plus verification')
    workers=min(workers,population_size)
    verify_count=min(2,population_size)
    rounds=min(generations,max(1,max_evaluations//(population_size+verify_count)))
    plan=schedule(mission,rounds)
    folder=Path(output); folder.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(seed)
    population=seed_population(population_size,mission,bounds,rng)
    if seed_policy:
        previous=load_policy(seed_policy)
        if previous.get('kind') not in ('guided_hybrid_v1','guided_hybrid_v2'):
            raise ValueError('--seed-policy requires a guided hybrid artifact')
        old=previous['network']; selected=previous['metadata']['selected_mission']
        epoch=float(spice.str2et(selected['epoch']))
        if not bounds[0]<=epoch<=bounds[1]:
            raise ValueError('Seed policy epoch is outside the requested window')
        population[0]=np.clip(np.r_[(epoch-bounds[0])/max(1,bounds[1]-bounds[0]),
            selected['departure_phase_deg']/360,(old.switch_perihelion_au-.12)/.22,
            (old.brake_cone_deg-25)/20,(old.reset_radius_au-.7)/2.3,
            getattr(old,'targeting_gain',0)/3,old.weights.ravel()/1.2+.5],0,1)
        print(f'Warm start from {seed_policy}; other candidates are freshly seeded.',flush=True)
    epoch_anchors=population[:,0].copy()
    metadata=dict(kind='guided_hybrid_v2',mission=asdict(mission),search=asdict(settings),seed_policy=str(seed_policy) if seed_policy else None,
        seed=seed,population=population_size,workers=workers,requested_generations=generations,
        rounds=rounds,max_evaluations=max_evaluations,features=FEATURES,
        epoch_start=spice.et2utc(bounds[0],'ISOC',3),epoch_end=spice.et2utc(bounds[1],'ISOC',3))
    metadata['versions']={name:importlib.metadata.version(name) for name in ('numpy','scipy','spiceypy')}
    metadata['source_sha256']={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ('main.py','guided_policy.py','optimizer.py','control.py','policy_io.py','search.py','dynamics.py','solar_sail.py','simulation.py',
                     'integrator.py','ephemerides.py','mission.py','fitness.py')}
    (folder/'search_request.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(f'Hybrid search: {population_size} candidates, {rounds} rounds, <= {max_evaluations} training rollouts',flush=True)
    print(f'Epoch interval: {metadata["epoch_start"]} through {metadata["epoch_end"]}',flush=True)
    print('Informed steering seeds with zero/small neural residuals, then joint date/steering refinement.',flush=True)
    history=[]; errors=[]; cache={}; count=0; pool=None; champion=None; previous_phase=None
    scores=np.full(population_size,-np.inf)
    targeting_round=0

    def run_jobs(vectors,phase,horizon,fidelity):
        nonlocal count
        reports={}; jobs=[]
        for i,v in enumerate(vectors):
            token=(phase,horizon,fidelity,np.asarray(v).tobytes())
            if token in cache:
                reports[i]=cache[token]
            elif count+len(jobs)<max_evaluations:
                jobs.append((i,v,phase,horizon,fidelity))
        count+=len(jobs)
        if pool:
            futures=[pool.submit(evaluate,job) for job in jobs]
            values=(f.result() for f in as_completed(futures))
        else:
            values=map(evaluate,jobs)
        for i,report,error in values:
            if error:
                errors.append(dict(candidate=i,phase=phase,fidelity=fidelity,error=error))
                print(f'candidate {i} failed: {error}',flush=True)
            else:
                cache[(phase,horizon,fidelity,np.asarray(vectors[i]).tobytes())]=report
                reports[i]=report
                closest=report['encounters'][-1]['closest_sample_au'] if report['encounters'] else None
                print(f'{phase}/{fidelity} {i}: epoch={report["epoch"][:10]}, '
                    f'score={report["search_score"]:.3f}, q={report["minimum_radius_au"]:.3f} AU, '
                    f'vinf={report["vinf_kms"]:.2f} km/s, miss={closest}',flush=True)
        if errors:
            (folder/'errors.json').write_text(json.dumps(errors,indent=2),encoding='utf-8')
        return reports

    try:
        if workers>1:
            pool=ProcessPoolExecutor(max_workers=workers,initializer=initialize_worker,
                initargs=(mission,settings,bounds,max(h for _,h in plan)))
        else:
            initialize_worker(mission,settings,bounds,max(h for _,h in plan))
        for generation,(phase,horizon) in enumerate(plan):
            transition=phase!=previous_phase
            if transition:
                champion=None
                trial=population.copy()
                scores[:]=-np.inf
                label=f'{horizon/86400:.1f} days' if horizon<YEAR else f'{horizon/YEAR:.1f} years'
                print(f'Round {generation+1}/{rounds}: {phase}, horizon {label}',flush=True)
            else:
                trial=population.copy()
                best=population[int(np.argmax(scores))]
                for i in range(population_size):
                    if population_size>=4:
                        others=np.delete(np.arange(population_size),i)
                        a,b=rng.choice(others,2,replace=False)
                        mutant=population[i]+.6*(best-population[i])+.7*(population[a]-population[b])
                    else:
                        mutant=population[i]+rng.normal(0,.12,DIMENSIONS)
                    mutant[PARAMETERS:]+=rng.normal(0,.025,DIMENSIONS-PARAMETERS)
                    mask=rng.random(DIMENSIONS)<.45
                    mask[rng.integers(DIMENSIONS)]=True
                    trial[i]=np.clip(np.where(mask,mutant,population[i]),0,1)
                # Occasionally explore a different date/phase without destroying the archive.
                if generation%3==2:
                    i=int(np.argmin(scores)); trial[i,:2]=rng.random(2)
                if phase=='scout':
                    trial[:,0]=epoch_anchors
                else:
                    # Coordinate shooting steps shrink from months to hours,
                    # while the remaining population continues global exploration.
                    days=180*(.45**targeting_round)
                    date_step=days*86400/max(1,bounds[1]-bounds[0])
                    for i,sign in enumerate((1,-1)):
                        if i<population_size:
                            trial[i]=best.copy(); trial[i,0]=np.clip(best[0]+sign*date_step,0,1)
                    if population_size>=4:
                        for i,sign in ((2,1),(3,-1)):
                            trial[i]=best.copy()
                            coordinate=PARAMETERS+len(FEATURES)+5
                            trial[i,coordinate]=np.clip(best[coordinate]+sign*.12*(.7**targeting_round),0,1)
                    targeting_round+=1
            screened=run_jobs(trial,phase,horizon,'screen')
            if not screened:
                raise RuntimeError('No screening rollouts completed; inspect errors.json or increase the budget')
            for i,r in screened.items():
                if r['search_score']>=scores[i]:
                    population[i]=trial[i]; scores[i]=r['search_score']
            ranked=np.argsort(-scores,kind='stable')
            choices=[int(ranked[0])]
            if verify_count>1:
                choices.append(int(ranked[1]) if generation%2==0 else int(rng.choice(ranked[1:])))
            checked=run_jobs(population[choices],phase,horizon,'accurate')
            for local in sorted(checked):
                r=checked[local]
                if champion is None or r['search_score']>champion['report']['search_score']:
                    champion=dict(vector=population[choices[local]].copy(),report=r)
            if champion is None:
                raise RuntimeError('No accurately checked policy; inspect errors.json or increase the budget')
            selected,network=decode(champion['vector'],mission,bounds)
            artifact=dict(kind='guided_hybrid_v2',network=network,vector=champion['vector'],
                metadata=dict(metadata,selected_mission=asdict(selected),trained_phase=phase,
                              trained_horizon=horizon,training_rollouts=count))
            for name in ('policy.pkl',f'policy_{phase}.pkl'):
                with (folder/name).open('wb') as stream:
                    pickle.dump(artifact,stream)
            np.savez_compressed(folder/'network.npz',weights=network.weights,features=np.array(FEATURES),
                switch_perihelion_au=network.switch_perihelion_au,brake_cone_deg=network.brake_cone_deg,
                reset_radius_au=network.reset_radius_au,epoch=selected.epoch,
                targeting_gain=network.targeting_gain,
                departure_phase_deg=selected.departure_phase_deg)
            history.append(dict(round=generation,phase=phase,horizon_years=horizon/YEAR,
                training_rollouts=count,epoch=selected.epoch,raw_score=champion['report']['search_score'],
                minimum_radius_au=champion['report']['minimum_radius_au'],vinf_kms=champion['report']['vinf_kms'],
                encounters=champion['report']['encounters']))
            (folder/'training_history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
            np.savez_compressed(folder/'population.npz',vectors=population,scores=scores)
            previous_phase=phase
    finally:
        if pool:
            pool.shutdown(wait=True,cancel_futures=True)
    # One final high-accuracy export is outside the training-rollout budget.
    selected,network=decode(champion['vector'],mission,bounds)
    label=f'{horizon/86400:.1f} days' if horizon<YEAR else f'{horizon/YEAR:.1f} years'
    print(f'Exact reference validation: selected epoch {selected.epoch}, {label}',flush=True)
    x0,base=setup(selected)
    params=rollout_parameters(base,settings,horizon,'reference')
    params['plot_dt']=min(params['plot_dt'],3600)
    params['progress_label']='hybrid final reference'
    result,report=rollout(network,selected,x0,params,phase=phase)
    report.update(epoch=selected.epoch,departure_phase_deg=selected.departure_phase_deg,
        optimizer='guided_hybrid_v2',search_phase=phase,full_mission_evaluated=phase=='mission' or report['feasible'],
        validation='reference_exact_spice',training_rollouts=count,requested_mission=asdict(mission))
    if phase!='mission' and not report['feasible']:
        report['feasible']=False
        report['status']='partial_horizon_candidate'
    save_result(folder,result,report)
    (folder/'selected_mission.json').write_text(json.dumps(asdict(selected),indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    return result,report,metadata


def replay(path,animate=False,full=False):
    path=Path(path)
    artifact=load_policy(path)
    metadata=artifact['metadata']; mission=Mission.from_saved(metadata['selected_mission'])
    settings=SearchSettings.from_saved(metadata['search'])
    settings.validate()
    x0,base=setup(mission)
    horizon=base['T'] if full else metadata['trained_horizon']
    phase='mission' if full else metadata['trained_phase']
    params=rollout_parameters(base,settings,horizon,'reference')
    params['plot_dt']=min(params['plot_dt'],3600)
    params['progress_label']='hybrid replay'
    result,report=rollout(artifact['network'],mission,x0,params,phase=phase)
    report.update(epoch=mission.epoch,departure_phase_deg=mission.departure_phase_deg,
                  optimizer=artifact['kind'],search_phase=phase,
                  full_mission_evaluated=phase=='mission' or report['feasible'],validation='reference_exact_spice',
                  training_rollouts=metadata.get('training_rollouts'),requested_mission=metadata['mission'])
    if phase!='mission' and not report['feasible']:
        report['feasible']=False; report['status']='partial_horizon_candidate'
    save_result(path.parent,result,report)
    if animate:
        from plotting import animate_rollout
        animate_rollout(result,base['et0'],path.parent/'animation.html')
    print(json.dumps(report,indent=2))
    return result,report
