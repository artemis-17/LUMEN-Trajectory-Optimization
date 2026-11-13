from pathlib import Path
import numpy as np
import spiceypy as spice

KERNEL_DIR = Path(__file__).resolve().parent / "kernels"
PLANET_IDS = tuple(range(1, 9))  # Mercury, Venus, Earth-Moon barycenter, Mars, ..., Neptune


def load_kernels():
    for name in ("naif0012.tls", "de440s.bsp", "gm_de440.tpc"):
        path = KERNEL_DIR / name
        if not path.is_file():
            raise FileNotFoundError(f"Download {name} into {KERNEL_DIR}")
        spice.furnsh(str(path))


def gm_si(body_id):
    _, values = spice.bodvcd(body_id, "GM", 1)
    return float(values[0]) * 1e9  # km^3/s^2 -> m^3/s^2


def planet_positions(et):
    # Geometric heliocentric positions, J2000; m
    return np.array([
        spice.spkpos(str(body_id), et, "J2000", "NONE", "SUN")[0]
        for body_id in PLANET_IDS
    ]) * 1e3


def earth_state(et):
    state, _ = spice.spkezr("EARTH", et, "J2000", "NONE", "SUN")
    return np.asarray(state) * 1e3  # m, m/s
