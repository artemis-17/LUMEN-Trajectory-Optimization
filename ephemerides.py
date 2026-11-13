from pathlib import Path
from functools import lru_cache
from contextlib import contextmanager
from scipy.interpolate import CubicHermiteSpline
import numpy as np
import spiceypy as spice

KERNEL_DIR = Path(__file__).resolve().parent / "kernels"
PLANET_IDS = tuple(range(1, 9))  # Mercury, Venus, Earth-Moon barycenter, Mars, ..., Neptune
NAMES = ('mercury', 'venus', 'earth', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune')
_force_table = None
_force_position = None


def load_kernels():
    exact_planet_positions.cache_clear()
    body_state.cache_clear()
    for name in ("naif0012.tls", "de440s.bsp", "gm_de440.tpc"):
        path = KERNEL_DIR / name
        if not path.is_file():
            raise FileNotFoundError(f"Download {name} into {KERNEL_DIR}")
        spice.furnsh(str(path))


def gm_si(body_id):
    _, values = spice.bodvcd(body_id, "GM", 1)
    return float(values[0]) * 1e9  # km^3/s^2 -> m^3/s^2


@lru_cache(maxsize=65536)
def exact_planet_positions(et):
    # Geometric heliocentric positions, J2000; m
    positions = np.array([
        spice.spkpos(str(body_id), et, "J2000", "NONE", "SUN")[0]
        for body_id in PLANET_IDS
    ]) * 1e3
    positions.setflags(write=False)
    return positions


def set_force_position(position):
    global _force_position
    _force_position = position


def planet_positions(et):
    """Exact by default; optionally interpolate forces away from close encounters."""
    if _force_table is None or _force_position is None:
        return exact_planet_positions(et)
    positions = _force_table.positions(et)
    if np.min(np.linalg.norm(positions-_force_position, axis=1)) < _force_table.guard_m:
        _force_table.exact_queries += 1
        return exact_planet_positions(et)
    _force_table.interpolated_queries += 1
    return positions


def near_planet(et, position, guard_au):
    positions = _force_table.positions(et) if _force_table is not None else exact_planet_positions(et)
    return bool(np.min(np.linalg.norm(positions-position, axis=1)) < guard_au*149597870700.0)


@contextmanager
def force_ephemeris(table=None):
    """Process-local rollout scope; guidance and fitness always query exact SPICE."""
    global _force_table, _force_position
    previous = _force_table, _force_position
    _force_table, _force_position = table, None
    try:
        yield
    finally:
        _force_table, _force_position = previous


class EphemerisTable:
    """Position/velocity-constrained cubic Hermite table; no extrapolation."""
    def __init__(self, et0, duration, step_days=0.5, guard_au=0.05, max_error_m=5000):
        if duration <= 0 or step_days <= 0 or guard_au <= 0 or max_error_m <= 0:
            raise ValueError('Ephemeris table parameters must be positive')
        self.et0, self.duration = et0, duration
        self.guard_m = guard_au*149597870700.0
        self.exact_queries = self.interpolated_queries = 0
        grid = np.linspace(0, duration, max(2, int(np.ceil(duration/(step_days*86400)))+1))
        states = np.array([[body_state(name, et0+float(t)) for name in NAMES] for t in grid])
        self.spline = CubicHermiteSpline(grid, states[:, :, :3], states[:, :, 3:], axis=0,
                                         extrapolate=False)
        # Independent off-node spot checks, including early/late intervals.
        selected = np.unique(np.linspace(0, len(grid)-2, min(64, len(grid)-1)).astype(int))
        checks = np.array([grid[i]+fraction*(grid[i+1]-grid[i])
                           for i in selected for fraction in (0.25, 0.5, 0.75)])
        errors = [np.linalg.norm(self.positions(et0+float(t))-exact_planet_positions(et0+float(t)), axis=1)
                  for t in checks]
        self.audit = dict(max_position_error_m=float(np.max(errors)), checks=len(checks),
                          nodes=len(grid), step_days=step_days, error_limit_m=max_error_m)
        if self.audit['max_position_error_m'] > max_error_m:
            raise ValueError(f'Ephemeris interpolation audit failed: {self.audit}')

    def positions(self, et):
        t = float(et-self.et0)
        if t < -1e-5 or t > self.duration+1e-5:
            raise ValueError('Ephemeris table queried outside its coverage')
        return self.spline(np.clip(t, 0, self.duration))


@lru_cache(maxsize=131072)
def body_state(name, et):
    """Planet-system barycenter heliocentric state, J2000, SI units."""
    state = np.asarray(spice.spkezr(str(NAMES.index(name) + 1), float(et),
                                  'J2000', 'NONE', 'SUN')[0]) * 1000
    state.setflags(write=False)
    return state
