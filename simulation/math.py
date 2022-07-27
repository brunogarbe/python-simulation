from .core import BasicBlock
from scipy.interpolate import interp1d

import numpy as np


class Gain(BasicBlock):
    
    def __init__(self, gain=[1], name=None):
        super().__init__(name=name)

        self.is_direct_feedthrough = True
        
        for _ in range(len(gain)):
            self.add_input()
            self.add_output()
            
        self.gain = np.array(gain)

    def output(self, t, x, u):
        return self.gain * u


class Sum(BasicBlock):
    
    def __init__(self, coef=[1, -1], name=None):
        super().__init__(name=name)

        self.is_direct_feedthrough = True

        for _ in range(len(coef)):
            self.add_input()
        self.add_output('output')

        self.coefs = np.array(coef)

    def output(self, t, x, u):
        return np.sum(self.coefs * u)

class Interpolate1D(BasicBlock):
    
    def __init__(self, x, y, kind='linear', name=None):
        super().__init__(name=name)

        self.is_direct_feedthrough = True

        self.add_input()
        self.add_output('output')

        self.interp_func = interp1d(x, y, kind='cubic')

    def output(self, t, x, u):
        return self.interp_func(u)