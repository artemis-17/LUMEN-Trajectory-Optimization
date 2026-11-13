import numpy as np
from solar_sail import AU

def make_neat_control(net):

    def control(t,x):

        r = x[:3]
        v = x[3:]

        # Calculate radial distance
        r_norm = np.linalg.norm(r)

        # Calculate radial velocity
        v_r = np.dot(v, r) / r_norm

        # Calculate transverse velocity
        v_t = np.linalg.norm(v - v_r * r / r_norm)

        # Normalize the inputs
        v_ref = 30000.0
        T_ref = 365.25 * 86400.0

        inputs = [
            r_norm / AU,
            v_r / v_ref,
            v_t / v_ref,
            t / T_ref
        ]

        # Evaluate the neural network
        y1, y2 = net.activate(inputs)
        alpha = (np.pi / 2) * y1
        delta = np.pi * (2 * y2 - 1)
        return alpha, delta

    return control