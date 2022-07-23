from .core import BasicBlock #, create_channels

from enum import Enum
import numpy as np


class Constant(BasicBlock):
  
    def __init__(self, value=1.0, name=None):
        super().__init__(name=name)
        
        # TODO: code for multiple outputs

        self.value = value
        
        self.add_output('output')

    def output(self, t, x, u):
        return self.value
            

# https://www.mathworks.com/help/simulink/slref/step.html
class Step(BasicBlock):
    """Make the extension
    This hook *function* gets picked up by the markdown processor
    when the extension is listed
    ```python
    output = markdown.markdown(
        content, extensions=[
            "admonition",
            "codehilite",
            "jetblack_markdown.autodoc",
        ])
    print(output)
    ```
    Returns:
        Extension: The extension
    """
  
    def __init__(self, fv=1.0, iv=0.0, ts=0.0, name=None):
        super().__init__(name=name)
        
        # TODO: code for multiple outputs

        self.ts = ts
        self.iv = iv
        self.fv = fv
        self.value = iv
        
        self.add_output('output')

    def output(self, t, x, u):
        return self.value
            
    def add_schedule(self, scheduler, t_start, t_final):
        if self.ts <= t_final:
            scheduler.add(self.ts, self.event_handler)

    def event_handler(self, x, u):
        self.value = self.fv


# https://en.wikipedia.org/wiki/Pulse_wave
class Pulse(BasicBlock):
  
    def __init__(self, T, tau=0.5, fv=1.0, iv=0.0, name=None):
        super().__init__(name=name)
        
        self.T = T
        self.fsm = False
        self.iv = iv
        self.fv = fv
        self.value = iv
        
        self.add_output('output')

    def output(self, t, x, u):
        return self.value
            
    def add_schedule(self, scheduler, t_final):
        t = 0.0
        while t < t_final:
            scheduler.add(t, self.event_handler)
            t += self.T/2

    def event_handler(self, x, u):
        if self.fsm == False:
            self.fsm = True
            self.value = self.fv
        else:
            self.fsm = False
            self.value = self.iv


# https://www.mathworks.com/help/simulink/slref/step.html
# Sine Wave
class Sine(BasicBlock):
    
    def __init__(self, fv=1.0, ws=0.0):
        super().__init__()
        
        self.ws = ws
        self.fv = fv
        self.add_output('output')

    def output(self, t, x, u):
        return np.sin(np.array([2*self.ws*t]))


# PMW signal
# sawtooth Ramp
# From Function
# InterpolateSignal (repeat and not repeated) # From Variable
# Chirp Signal
# Constant
# Band-Limited White Noise
# Random Number
# Uniform Random Number
