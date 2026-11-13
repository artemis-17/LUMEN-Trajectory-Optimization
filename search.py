"""Search defaults and propagation accuracy profiles; no algorithm-specific gates."""
from dataclasses import dataclass,fields
import numpy as np


@dataclass
class SearchSettings:
    population: int = 12
    generations: int = 100
    workers: int = 4
    max_evaluations: int = 240
    seed: int = 42
    screen_rtol: float = 1e-6
    screen_position_atol_m: float = 100.0
    screen_velocity_atol_mps: float = 1e-3
    screen_sample_hours: float = 24.0
    screen_guard_au: float = 0.1
    interpolate: bool = True
    ephemeris_step_days: float = 0.5
    ephemeris_guard_au: float = 0.05
    ephemeris_error_limit_m: float = 5000.0

    @classmethod
    def from_saved(cls,payload):
        # Older guided-policy artifacts contain retired curriculum settings.
        # Preserve their active accuracy fields without reviving old algorithms.
        names={f.name for f in fields(cls)}
        return cls(**{k:v for k,v in payload.items() if k in names})

    def validate(self):
        for name in ('population','generations','workers','max_evaluations'):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,int) or value<1:
                raise ValueError(f'{name} must be a positive integer')
        if isinstance(self.seed,bool) or not isinstance(self.seed,int) or self.seed<0:
            raise ValueError('seed must be a nonnegative integer')
        for name in ('screen_rtol','screen_position_atol_m','screen_velocity_atol_mps',
                     'screen_sample_hours','screen_guard_au','ephemeris_step_days',
                     'ephemeris_guard_au','ephemeris_error_limit_m'):
            if not np.isfinite(getattr(self,name)) or getattr(self,name)<=0:
                raise ValueError(f'{name} must be finite and positive')
        if not isinstance(self.interpolate,bool):
            raise ValueError('interpolate must be true or false')


def rollout_parameters(base,settings,horizon,fidelity):
    """Copy the physical scenario; change numerical accuracy for screening only."""
    params=dict(base,T=horizon,mission_T=base['T'],backend='persistent')
    if fidelity=='screen':
        params.update(rtol=settings.screen_rtol,
            atol=[settings.screen_position_atol_m]*3+[settings.screen_velocity_atol_mps]*3,
            plot_dt=max(base['plot_dt'],settings.screen_sample_hours*3600),
            close_guard_au=settings.screen_guard_au,close_rtol=base['rtol'],close_atol=base['atol'])
    elif fidelity=='reference':
        params['backend']='reference'
    elif fidelity!='accurate':
        raise ValueError(f'Unknown fidelity: {fidelity}')
    return params
