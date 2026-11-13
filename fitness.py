"""Trajectory metrics, ordered sampled encounters and the search objective."""
import numpy as np
from mission import AU,DAY
from ephemerides import body_state


def assess(result,mission,params):
    """Only propagated samples establish encounters; forecasts never certify hits."""
    t=np.asarray(result['t']); X=np.asarray(result['X']).T
    encounters=[]; start=0; completed=0; predicted=None
    for index,name in enumerate(mission.targets):
        if start>=len(t):
            distance=np.linalg.norm(body_state(name,params['et0']+t[-1])[:3]-X[-1,:3])
            encounters.append(dict(planet=name,closest_sample_au=float(distance/AU),
                closest_sample_day=float(t[-1]/DAY),hit=False,observed_after_previous=False))
            break
        states=np.array([body_state(name,params['et0']+float(ti)) for ti in t[start:]])
        distances=np.linalg.norm(states[:,:3]-X[start:,:3],axis=1)
        closest=int(np.argmin(distances)); hits=np.flatnonzero(distances<=mission.flyby_radius_au*AU)
        encounters.append(dict(planet=name,closest_sample_au=float(distances[closest]/AU),
            closest_sample_day=float(t[start+closest]/DAY),hit=bool(len(hits))))
        if len(hits):
            completed+=1
            start+=int(hits[0])+1
            continue
        records=np.asarray(result.get('predictions',[]),dtype=float).reshape(-1,4)
        if len(records):
            mask=(records[:,1]==index)&(records[:,0]>=t[start])
            eligible=records[mask]
            if len(eligible):
                present=np.interp(eligible[:,0],t[start:],distances)
                improvement=np.maximum(0,1/(1+eligible[:,3]/AU)-1/(1+present/AU))
                weights=improvement/(1+eligible[:,2]/(365.25*DAY))
                best=int(np.argmax(weights))
                models=np.asarray(result.get('prediction_models',['unknown']*len(records)))
                predicted=dict(planet=name,evaluated_day=float(eligible[best,0]/DAY),
                    time_to_cpa_days=float(eligible[best,2]/DAY),
                    predicted_miss_au=float(eligible[best,3]/AU),model=str(models[mask][best]))
        break
    xf=result['x_final']; radius=np.linalg.norm(xf[:3]); mu=params['mu_sun']
    energy=float(np.dot(xf[3:],xf[3:])/2-mu/radius)
    initial_energy=float(np.dot(result['x0'][3:],result['x0'][3:])/2-mu/np.linalg.norm(result['x0'][:3]))
    vinf=float(np.sqrt(max(0,2*energy))/1000)
    outbound=bool(np.dot(xf[:3],xf[3:])>0)
    feasible=bool(result['status']=='reached_goal' and completed==len(mission.targets)
                  and outbound and energy>0 and vinf>=mission.min_vinf_kms)
    peri_time=result.get('t_r_min',float(t[np.argmin(np.linalg.norm(X[:,:3],axis=1))]))
    peri_state=result.get('x_r_min')
    if peri_state is None:
        peri_state=np.array([np.interp(peri_time,t,row) for row in result['X']])
    peri_energy=float(np.dot(peri_state[3:],peri_state[3:])/2-mu/np.linalg.norm(peri_state[:3]))
    return dict(status='feasible_in_model' if feasible else 'no_solution_found',feasible=feasible,
        horizon_days=float(params['T']/DAY),encounters=encounters,completed=completed,
        minimum_radius_au=float(result['r_min']/AU),perihelion_day=float(peri_time/DAY),
        perihelion_energy_si=peri_energy,post_perihelion_energy_gain_si=energy-peri_energy,
        post_perihelion_days=float(max(0,result['t_final']-peri_time)/DAY),
        initial_energy_si=initial_energy,final_energy_si=energy,vinf_kms=vinf,outbound=outbound,
        predicted_encounter=predicted,flight_days=float(result['t_final']/DAY),
        propagation_status=result['status'],nfev=result.get('nfev'),
        caveat='Sampled barycenter encounters; no Moon, shadows, planet surfaces or attitude constraints. Search failure is not proof of infeasibility.')


def score_trajectory(report,result,phase,mission):
    if report['propagation_status']=='solar_limit':
        return -10000.0
    radius=np.linalg.norm(result['x_final'][:3])/AU
    next_distance=(report['encounters'][-1]['closest_sample_au']
                   if report['encounters'] else mission.escape_radius_au)
    # Smooth, high-dynamic-range miss-distance shaping; near misses remain useful.
    encounter=(-100*np.log10(1+next_distance/mission.flyby_radius_au)
               if report['completed']<len(mission.targets) else 0.0)
    escape=10*np.tanh(report['vinf_kms']/20)+15*np.clip(radius/mission.escape_radius_au,0,1)
    if phase=='scout':
        initial_energy=report['initial_energy_si']
        approach=20/(1+abs(report['minimum_radius_au']-.24)/.15)
        return float(approach+15*np.tanh((report['final_energy_si']-initial_energy)/4.5e8)
                     +escape+.15*encounter)
    science_complete=report['completed']==len(mission.targets)
    score=encounter+(escape if science_complete else .25*escape)+300*report['completed']
    if science_complete and report['propagation_status']=='reached_goal':
        score+=25
    if report['feasible']:
        score+=1000+50*(1-report['flight_days']/(mission.years*365.25))
    return float(score)
