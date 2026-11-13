import numpy as np
from solar_sail import AU

def control(t, x):

    r = np.linalg.norm(x[:3])

    alpha = np.deg2rad(35.0)

    if r > 0.2 * AU:
        delta = np.pi  # Retrograde
    else:
        delta = 0.0    # Prograde

    return alpha, delta