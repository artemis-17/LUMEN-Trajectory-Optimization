"""Mission configuration, ephemeris features and initial conditions."""
from dataclasses import dataclass, field,fields
import numpy as np
import spiceypy as spice
from ephemerides import load_kernels, gm_si, PLANET_IDS, NAMES, body_state
from solar_sail import AU

DAY = 86400.0
YEAR = 365.25 * DAY
@dataclass
class Mission:
    epoch: str = '2035-01-01T00:00:00'
    epoch_start: str | None = None
    epoch_end: str | None = None
    years: float = 30.0
    targets: list = field(default_factory=lambda: ['jupiter'])
    flyby_radius_au: float = 0.02
    escape_radius_au: float = 100.0
    min_vinf_kms: float = 20.0
    departure_radius_km: float = 100000.0
    departure_phase_deg: float = 0.0
    departure_inclination_deg: float = 0.0
    control_days: float = 1.0
    sample_hours: float = 6.0
    area: float = 340.0
    mass: float = 5.0
    reflectivity: float = 0.9
    solar_limit_au: float = 0.2

    @classmethod
    def from_saved(cls,payload):
        names={f.name for f in fields(cls)}
        return cls(**{k:v for k,v in payload.items() if k in names})

    def validate(self):
        for key in ('years', 'flyby_radius_au', 'escape_radius_au',
                    'departure_radius_km', 'control_days', 'sample_hours', 'area', 'mass', 'solar_limit_au'):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f'{key} must be finite and positive')
        if not all(np.isfinite(v) for v in (self.min_vinf_kms, self.reflectivity,
                                            self.departure_phase_deg, self.departure_inclination_deg)):
            raise ValueError('Mission scalar parameters must be finite')
        if self.escape_radius_au <= 1.1 or self.min_vinf_kms < 0:
            raise ValueError('Escape radius must exceed 1.1 AU; min_vinf must be nonnegative')
        if not 0 <= self.reflectivity <= 1 or self.departure_radius_km < 10000:
            raise ValueError('Invalid reflectivity or departure radius (minimum 10000 km)')
        self.targets = [p.lower() for p in self.targets]
        if any(p not in NAMES for p in self.targets):
            raise ValueError(f'Targets must be chosen from {NAMES}')


def setup(mission,initialize=True):
    mission.validate()
    if initialize:
        load_kernels()
    et0 = spice.str2et(mission.epoch)
    # Check both endpoints before launching a costly population search.
    for name in NAMES:
        body_state(name, et0)
        body_state(name, et0 + mission.years * YEAR)
    earth = body_state('earth', et0)
    er = earth[:3] / np.linalg.norm(earth[:3])
    eh = np.cross(earth[:3], earth[3:]); eh /= np.linalg.norm(eh)
    et = np.cross(eh, er)
    phase, inc = np.deg2rad([mission.departure_phase_deg, mission.departure_inclination_deg])
    tangent = np.cos(inc)*et + np.sin(inc)*eh
    radial = np.cos(phase)*er + np.sin(phase)*tangent
    along = -np.sin(phase)*er + np.cos(phase)*tangent
    radius = mission.departure_radius_km * 1000
    x0 = earth + np.r_[radius*radial, np.sqrt(gm_si(3)/radius)*along]
    params = dict(et0=et0, mu_sun=gm_si(10),
                  mu_planets=np.array([gm_si(i) for i in PLANET_IDS]),
                  sail_params=(mission.area, mission.mass, mission.reflectivity),
                  T=mission.years*YEAR, r_min=mission.solar_limit_au*AU,
                  r_goal=mission.escape_radius_au*AU,
                  dt_control=mission.control_days*DAY, max_step=DAY,
                  plot_dt=mission.sample_hours*3600, rtol=1e-10,
                  atol=[0.1]*3 + [1e-6]*3)
    return x0, params
