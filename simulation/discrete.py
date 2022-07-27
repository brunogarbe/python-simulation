# Delay	Delay input signal by fixed or variable sample periods
# Difference	Calculate change in signal over one time step
# Discrete Derivative	Compute discrete-time derivative
# Discrete FIR Filter	Model FIR filters
# Discrete Filter	Model Infinite Impulse Response (IIR) filters
# Discrete PID Controller	Discrete-time or continuous-time PID controller
# Discrete PID Controller (2DOF)	Discrete-time or continuous-time two-degree-of-freedom PID controller
# Discrete State-Space	Implement discrete state-space system
# Discrete Transfer Fcn	Implement discrete transfer function
# Discrete Zero-Pole	Model system defined by zeros and poles of discrete transfer function
# Discrete-Time Integrator	Perform discrete-time integration or accumulation of signal
# First-Order Hold (Obsolete)	Implement first-order sample-and-hold
# Memory	Output input from previous time step
# Resettable Delay	Delay input signal by variable sample period and reset with external signal
# Tapped Delay	Delay scalar signal multiple sample periods and output all delayed versions
# Transfer Fcn First Order	Implement discrete-time first order transfer function
# Transfer Fcn Lead or Lag	Implement discrete-time lead or lag compensator
# Transfer Fcn Real Zero	Implement discrete-time transfer function that has real zero and no pole
# Unit Delay	Delay signal one sample period
# Variable Integer Delay	Delay input signal by variable sample period
# Zero-Order Hold	Implement zero-order hold sample period

from .core import BasicBlock

import numpy as np

# Discontinuities
# https://lost-contact.mit.edu/afs/inf.ed.ac.uk/group/teaching/matlab-help/R2016b/simulink/slref/saturation.html
# Saturation
# Saturation Dynamic

# Saturation	Limit range of signal
# Backlash	Model behavior of system with play
# Coulomb and Viscous Friction	Model discontinuity at zero, with linear gain elsewhere
# Dead Zone	Provide region of zero output
# Dead Zone Dynamic	Set inputs within bounds to zero
# Hit Crossing	Detect crossing point
# Quantizer	Discretize input at specified interval
# Rate Limiter	Limit rate of change of signal
# Rate Limiter Dynamic	Limit rising and falling rates of signal
# Relay	Switch output between two constants
# Saturation Dynamic	Bound range of input
# Wrap To Zero	Set output to zero if input is above threshold

# Backlash	Model behavior of system with play
# Coulomb and Viscous Friction	Model discontinuity at zero, with linear gain elsewhere
# Dead Zone	Provide region of zero output
# Dead Zone Dynamic	Provide dynamic region of zero output
# Hit Crossing	Detect crossing point
# PWM	Generate ideal pulse width modulated signal corresponding to input duty cycle
# Quantizer	Discretize input at given interval
# Rate Limiter	Limit rate of change of signal
# Rate Limiter Dynamic	Limit rate of change of signal
# Relay	Switch output between two constants
# Saturation	Limit input signal to the upper and lower saturation values
# Saturation Dynamic	Limit input signal to dynamic upper and lower saturation values
# Variable Pulse Generator	Generate ideal, time varying pulse signal
# Wrap To Zero	Set output to zero if input is above threshold

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
        # if self.fsm == True:
        #     self.state = self.difference(0, self.state, u)
        #     self.value = u[0]
        #     self.fsm = False
        
        return np.array([self.state])

    def difference(self, n, x, u):
        xn = 0.9*x[0]
        return np.array([xn])

            
    def add_schedule(self, scheduler, t_start, t_final):
        dt = 0.5
        samples = np.arange(dt, t_final + 0.001, dt)
        #print('samples', samples)
        for s in samples:
            scheduler.add(s, self.test_handler)

    def test_handler(self, t, x, u):
        #self.fsm = True
        #print('================================================')
        #print('x', x)
        #print('u', u)
        self.state = self.difference(0, self.state, u)
        #print('================================================')