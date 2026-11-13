"""Stateful sail guidance: departure, braking, pumping and target steering."""
from time import perf_counter
import numpy as np
from ephemerides import body_state
from mission import AU,DAY,YEAR
from encounters import gravity_closest_approach,lambert_departure_velocity

def work_maximizing_cone(radial_speed,transverse_speed,rho):
    """Choose positive-work steering from analytic/nearby and edge candidates."""
    vt = max(float(transverse_speed),1e-12)
    ideal = np.arctan2(-3*radial_speed+np.sqrt(9*radial_speed**2+8*vt**2),4*vt)
    angles = np.clip(np.r_[0,np.pi/2,ideal+np.deg2rad([-15,-7,0,7,15])],0,np.pi/2)
    c,s = np.cos(angles),np.sin(angles)
    work = radial_speed*((1-rho)*c+2*rho*c**3)+transverse_speed*2*rho*c*c*s
    return float(angles[int(np.argmax(work))])


def osculating_perihelion(x,mu):
    r,v = x[:3],x[3:]
    h = np.cross(r,v)
    eccentricity = np.linalg.norm(np.cross(v,h)/mu-r/np.linalg.norm(r))
    return float(np.dot(h,h)/(mu*(1+eccentricity)))


class GuidedControl:
    def __init__(self,network,mission,params):
        self.network,self.mission,self.params = network,mission,params
        self.index=0
        self.mode='departure'
        self.commands=[]; self.predictions=[]; self.prediction_models=[]
        self.last_progress=perf_counter()
        self.forecast=None

    def __call__(self,t,x):
        p,m,n = self.params,self.mission,self.network
        if p.get('progress_label') and perf_counter()-self.last_progress > 10:
            print(f"{p['progress_label']}: day {t/DAY:.1f}/{p['T']/DAY:.1f}, {self.mode}",flush=True)
            self.last_progress=perf_counter()
        r,v=x[:3],x[3:]
        radius=np.linalg.norm(r); er=r/radius
        eh=np.cross(r,v); eh/=np.linalg.norm(eh)
        et=np.cross(eh,er)
        basis=np.array([er,et,eh])
        earth=body_state('earth',p['et0']+t)
        earth_distance=np.linalg.norm(r-earth[:3])
        target=None
        if self.index < len(m.targets):
            target=body_state(m.targets[self.index],p['et0']+t)
            if np.linalg.norm(target[:3]-r) <= m.flyby_radius_au*AU:
                self.index+=1
                target=body_state(m.targets[self.index],p['et0']+t) if self.index<len(m.targets) else None
        if self.mode=='departure' and earth_distance > 0.015*AU:
            self.mode='inbound'
        perihelion=osculating_perihelion(x,p['mu_sun'])
        if self.mode=='inbound' and perihelion <= n.switch_perihelion_au*AU:
            self.mode='energy'
        solar_energy=np.dot(v,v)/2-p['mu_sun']/radius
        if (self.mode=='energy' and solar_energy < 0 and radius > n.reset_radius_au*AU
                and perihelion > 1.15*n.switch_perihelion_au*AU):
            self.mode='reset'
        if self.mode=='reset' and (perihelion <= n.switch_perihelion_au*AU or solar_energy >= 0):
            self.mode='energy'
        radial=float(np.dot(v,er))
        transverse=float(np.dot(v,et))
        if self.mode=='departure':
            local=basis@(v-earth[3:])
            alpha=work_maximizing_cone(local[0],np.hypot(local[1],local[2]),m.reflectivity)
            clock=float(np.arctan2(local[2],local[1]))
        elif self.mode in ('inbound','reset'):
            alpha=np.deg2rad(n.brake_cone_deg)
            clock=np.pi
        else:
            desired=np.array([radial,transverse,0.0])/max(np.linalg.norm(v),1)
            if n.targeting_gain>0 and target is not None and radius>.5*AU and (solar_energy>0 or radial>0):
                update_days=max(1,min(30,14*(radius/AU)**1.5))
                refresh=(self.forecast is None or t-self.forecast['at']>=update_days*DAY
                         or self.forecast['index']!=self.index)
                if refresh:
                    try:
                        tau,miss,vector=gravity_closest_approach(x,m.targets[self.index],p['et0']+t,
                            p['mu_sun'],min(3*YEAR,max(0,p.get('mission_T',p['T'])-t)),24)
                        self.forecast=dict(at=t,index=self.index,arrival=t+tau,miss=miss,vector=vector)
                        tof=min(2.5*YEAR,max(2*DAY,np.linalg.norm(target[:3]-r)/20000),
                                max(1,p.get('mission_T',p['T'])-t))
                        future=body_state(m.targets[self.index],p['et0']+t+tof)
                        try:
                            self.forecast['velocity']=lambert_departure_velocity(r,future[:3],tof,p['mu_sun'],eh)
                        except ValueError:
                            pass
                    except (ValueError,FloatingPointError):
                        self.forecast=None
                if self.forecast is not None and self.forecast['arrival']>t:
                    miss=self.forecast['miss']; vector=self.forecast['vector']
                    desired+=n.targeting_gain*(basis@vector)/(AU+miss)
                    self.predictions.append((float(t),self.index,self.forecast['arrival']-t,float(miss)))
                    self.prediction_models.append('guided_solar_coast')
                if self.forecast is not None and 'velocity' in self.forecast:
                    # Aim thrust along the remaining Lambert velocity error,
                    # blended with energy growth. No impulsive delta-v is applied.
                    dv=basis@(self.forecast['velocity']-v)
                    desired=np.array([radial,transverse,0.0])/max(np.linalg.norm(v),1)
                    desired+=n.targeting_gain*dv/(10000+np.linalg.norm(dv))
            alpha=work_maximizing_cone(desired[0],np.hypot(desired[1],desired[2]),m.reflectivity)
            clock=float(np.arctan2(desired[2],desired[1]))
        direction=np.zeros(3) if target is None else basis@(target[:3]-r)
        direction=direction/(AU+np.linalg.norm(direction))
        energy=np.dot(v,v)/2-p['mu_sun']/radius
        features=[1,np.clip(np.log(radius/AU),-3,5),np.tanh(radial/30000),
                  np.tanh(energy/4.5e8),direction[1],direction[2],
                  {'departure':-1,'inbound':0,'reset':0.5,'energy':1}[self.mode],
                  earth_distance/(earth_distance+0.01*AU)]
        da,dd=n.activate(features)
        alpha=float(np.clip(alpha+np.deg2rad(n.residual_cone_deg)*da,0,np.pi/2))
        clock=float((clock+np.deg2rad(n.residual_clock_deg)*dd+np.pi)%(2*np.pi)-np.pi)
        if self.mode=='energy' and radius<n.reset_radius_au*AU:
            clock=float(np.clip(clock,-np.pi/2,np.pi/2))
        # A command safeguard, not an altered force or a feasibility guarantee.
        if radius < 1.08*p['r_min'] and radial < 0:
            alpha,clock=0.0,0.0
        self.commands.append((float(t),alpha,clock))
        return alpha,clock
