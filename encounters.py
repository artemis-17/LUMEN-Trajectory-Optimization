"""Solar-gravity coast and Lambert shooting utilities.

These are guidance features/reward shaping, not replacements for the full
gravity/SRP propagation or evidence of completed flybys.
"""
import numpy as np
from scipy.optimize import minimize_scalar,brentq
from ephemerides import body_state


def _stumpff(z):
    """Stable universal-variable C/S functions (large trial brackets are capped)."""
    z = np.asarray(z,dtype=float)
    c, s = np.empty_like(z), np.empty_like(z)
    small = np.abs(z) < 1e-5
    c[small] = 0.5-z[small]/24+z[small]**2/720-z[small]**3/40320
    s[small] = 1/6-z[small]/120+z[small]**2/5040-z[small]**3/362880
    positive = z >= 1e-5
    u = np.sqrt(z[positive])
    c[positive] = (1-np.cos(u))/z[positive]
    s[positive] = (u-np.sin(u))/u**3
    negative = z <= -1e-5
    u = np.minimum(np.sqrt(-z[negative]),100)
    c[negative] = (np.cosh(u)-1)/(-z[negative])
    s[negative] = (np.sinh(u)-u)/(-z[negative])**1.5
    return c,s


def solar_coast_positions(state, times, mu):
    """Vectorized universal-variable two-body coast (no future sail commands).

    The safeguarded solve handles bound/unbound states. Only used for forecasts;
    spacecraft propagation still uses the full, unchanged gravity/SRP equations.
    """
    state, times = np.asarray(state,dtype=float),np.atleast_1d(times).astype(float)
    if mu <= 0 or np.any(times < 0) or not np.all(np.isfinite(times)):
        raise ValueError('Invalid coast parameters')
    r0 = np.linalg.norm(state[:3]); root_mu = np.sqrt(mu)
    if r0 <= 0 or not np.all(np.isfinite(state)):
        raise ValueError('Invalid initial coast state')
    alpha = 2/r0-np.dot(state[3:],state[3:])/mu
    radial_term = np.dot(state[:3],state[3:])/root_mu
    def equation(chi):
        z = alpha*chi*chi
        c,s = _stumpff(z)
        value = radial_term*chi*chi*c+(1-alpha*r0)*chi**3*s+r0*chi-root_mu*times
        derivative = radial_term*chi*(1-z*s)+(1-alpha*r0)*chi*chi*c+r0
        return value,derivative,c,s
    low = np.zeros_like(times)
    high = root_mu*times/r0+np.sqrt(r0)
    for _ in range(60):
        needs = equation(high)[0] < 0
        if not np.any(needs):
            break
        high[needs] *= 2
    else:
        raise ValueError('Could not bracket coast forecast')
    chi = (low+high)/2
    for _ in range(100):
        value,derivative,c,s = equation(chi)
        converged = np.abs(value) <= 1e-11*root_mu*np.maximum(times,1)
        if np.all(converged):
            break
        high = np.where((value > 0)&~converged,chi,high)
        low = np.where((value <= 0)&~converged,chi,low)
        proposal = chi-value/np.maximum(derivative,1e-30)
        proposal = np.where((proposal > low)&(proposal < high)&np.isfinite(proposal),
                            proposal,(low+high)/2)
        chi = np.where(converged,chi,proposal)
    else:
        raise ValueError('Coast forecast did not converge')
    _,_,c,s = equation(chi)
    f = 1-chi*chi*c/r0
    g = times-chi**3*s/root_mu
    positions = f[:,None]*state[:3]+g[:,None]*state[3:]
    positions[times == 0] = state[:3]
    return positions


def gravity_closest_approach(state, target, et, mu, horizon, samples=40):
    """Coarse Sun-gravity spacecraft coast vs actual moving SPICE target.

    Refine around the best sampled basin. This omits SRP/planetary perturbations,
    can miss other minima, and never certifies an encounter.
    """
    times = np.linspace(0,max(0,float(horizon)),max(3,int(samples)))
    spacecraft = solar_coast_positions(state,times,mu)
    planets = np.array([body_state(target,et+float(t))[:3] for t in times])
    vectors = planets-spacecraft
    distances = np.linalg.norm(vectors,axis=1)
    best = int(np.argmin(distances))
    tau,miss,vector = float(times[best]),float(distances[best]),vectors[best]
    if horizon > 0:
        left,right = times[max(0,best-1)],times[min(len(times)-1,best+1)]
        def objective(t):
            dr = body_state(target,et+float(t))[:3]-solar_coast_positions(state,[t],mu)[0]
            return float(np.dot(dr,dr))
        if right > left:
            refined = minimize_scalar(objective,bounds=(left,right),method='bounded',
                                      options=dict(xatol=3600,maxiter=24))
            if refined.success and refined.fun < miss*miss:
                tau = float(refined.x)
                vector = body_state(target,et+tau)[:3]-solar_coast_positions(state,[tau],mu)[0]
                miss = float(np.linalg.norm(vector))
    return tau,miss,vector


def lambert_departure_velocity(r1,r2,flight_time,mu,normal):
    """Zero-revolution Sun-only Lambert shooting velocity, not an applied impulse.

    Used to choose a continuous sail command; the spacecraft never receives this
    velocity instantaneously. Near-collinear/unsolved geometry raises ValueError.
    """
    r1,r2=np.asarray(r1,dtype=float),np.asarray(r2,dtype=float)
    a,b=np.linalg.norm(r1),np.linalg.norm(r2)
    cosine=float(np.clip(np.dot(r1,r2)/(a*b),-1,1))
    sine_sign=1 if np.dot(np.cross(r1,r2),normal)>=0 else -1
    A=sine_sign*np.sqrt(a*b*(1+cosine))
    if flight_time<=0 or abs(A)<1e-8*np.sqrt(a*b) or cosine>1-1e-10:
        raise ValueError('Degenerate Lambert geometry')
    def geometry(z):
        C,S=_stumpff(np.array([z])); C,S=float(C[0]),float(S[0])
        if C<=0:
            return None
        y=a+b+A*(z*S-1)/np.sqrt(C)
        if y<=0:
            return None
        time=((y/C)**1.5*S+A*np.sqrt(y))/np.sqrt(mu)
        return time,y
    grid=np.linspace(-4*np.pi**2,4*np.pi**2*(1-1e-7),100)
    previous=None; bracket=None
    for z in grid:
        value=geometry(float(z))
        if value is None:
            continue
        residual=value[0]-flight_time
        if previous is not None and previous[1]*residual<=0:
            bracket=(previous[0],float(z)); break
        previous=(float(z),residual)
    if bracket is None:
        raise ValueError('Lambert flight time not bracketed')
    z=brentq(lambda z:geometry(z)[0]-flight_time,*bracket,xtol=1e-10)
    y=geometry(z)[1]
    f=1-y/a; g=A*np.sqrt(y/mu)
    if abs(g)<1e-12:
        raise ValueError('Degenerate Lambert time coefficient')
    return (r2-f*r1)/g
