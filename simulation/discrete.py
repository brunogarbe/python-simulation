from .core import BasicBlock

import numpy as np

class ZeroOrderHold(BasicBlock):

    def __init__(self, x0=0.0, ts=0.0):
        super().__init__()

        self.value = x0
        self.fsm = False

        self.is_direct_feedthrough = True

        self.add_input('input')
        self.add_output('output')

    def output(self, t, x, u):
        if self.fsm == True:
            self.value = u[0]
            self.fsm = False
        
        return np.array([self.value])
            
    def add_schedule(self, scheduler, t_start, t_final):
        dt = 0.5
        samples = np.arange(dt, t_final + 0.001, dt)
        #print('samples', samples)
        for s in samples:
            scheduler.add(s, self.test_handler)

    def test_handler(self, t, x, u):
        self.fsm = True


class DiscreteTeste(BasicBlock):

    def __init__(self, x0=0.0, ts=0.0):
        super().__init__()

        self.value = x0
        self.fsm = False
        self.state = np.array([1.0])

        self.is_direct_feedthrough = True

        self.add_input('input')
        self.add_output('output')

    def output(self, t, x, u):
        return np.array([self.state])

    def difference(self, n, x, u):
        xn = 0.9*x[0]
        return np.array([xn])

    def add_schedule(self, scheduler, t_start, t_final):
        dt = 0.5
        samples = np.arange(dt, t_final + 0.001, dt)
        for s in samples:
            scheduler.add(s, self.test_handler)

    def test_handler(self, t, x, u):
        self.state = self.difference(0, self.state, u)
