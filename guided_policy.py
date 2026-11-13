"""Fixed two-output residual neural network and steering parameters.

The module/class name is retained so existing guided policy artifacts replay.
"""
from dataclasses import dataclass,field
import numpy as np

FEATURES = ['bias','log_solar_radius','radial_speed','solar_energy',
            'target_transverse','target_normal','mode','earth_distance']


@dataclass
class GuidedNetwork:
    switch_perihelion_au: float = 0.22
    brake_cone_deg: float = 35.264
    reset_radius_au: float = 1.0
    targeting_gain: float = 0.0
    weights: np.ndarray = field(default_factory=lambda:np.zeros((2,len(FEATURES))))
    residual_cone_deg: float = 12.0
    residual_clock_deg: float = 40.0

    def __post_init__(self):
        self.weights=np.asarray(self.weights,dtype=float)
        if self.weights.shape!=(2,len(FEATURES)) or not np.isfinite(self.weights).all():
            raise ValueError('Residual network must have finite 2 by 8 weights')
        if self.switch_perihelion_au<=0 or not 0<=self.brake_cone_deg<=90 or self.reset_radius_au<=0:
            raise ValueError('Invalid guided steering parameters')

    def activate(self,features):
        return np.tanh(self.weights@np.asarray(features))

    def make_control(self,mission,params):
        from control import GuidedControl
        return GuidedControl(self,mission,params)
