from .core import StateBlock

from enum import Enum
import numpy as np
from scipy.signal import tf2ss

class Integrator(StateBlock):

    def __init__(self, x0=0.0, name=None):
        super().__init__(name=name)

        self.num_states = 1

        self.add_input('input')
        self.add_output('output')

        # set the initial conditions
        self.x0 = np.array([x0])

    def initial(self):
        return self.x0

    # overriding abstract method
    def derivative(self, t, x, u):
        dxdt_0 = u[0]
        return np.array([dxdt_0])

    def output(self, t, x, u):
        return x

class FirstOrder(StateBlock):

    def __init__(self, x0=0.0, tau=1.0, dc_gain=1.0, name=None):
        super().__init__(name=name)

        self.num_states = 1

        self.add_input('input')
        self.add_output('output')

        # set the initial conditions
        self.x0 = np.array([x0])

        # set the parameters
        self.tau = tau
        self.K = dc_gain

        # TODO: check for zeros in the parameters!!!

    def initial(self):
        return self.x0

    # overriding abstract method
    def derivative(self, t, x, u):
        dxdt_0 = -(1.0/self.tau)*x[0] + (self.K/self.tau)*u[0]
        return np.array([dxdt_0])

    def output(self, t, x, u):
        return x

# SecondOrder

class StateSpace(StateBlock):

    def __init__(self, A, B, C, D=None, x0=None, name=None):
        super().__init__(name=name)

        self.num_states = A.shape[0]
        num_inputs = B.shape[1]
        num_outputs = C.shape[0]

        for i in range(num_inputs):
            self.add_input('input' + str(i))

        for i in range(num_outputs):
            self.add_output('output' + str(i))

        # set the initial conditions
        if x0 is None:
            self.x0 = np.zeros(self.num_states)

        self.A = np.array(A)
        self.B = np.array(B)
        self.C = np.array(C)
        
        if D is None or np.all( D == 0 ):
            self.is_direct_feedthrough = False
            self.D = None
        else:
            self.is_direct_feedthrough = True
            self.D = D


    def initial(self):
        return self.x0

    # overriding abstract method
    def derivative(self, t, x, u): #, u):
        xdot = np.matmul(self.A, np.reshape(x, (self.num_states, 1))) + np.matmul(self.B, np.reshape(u, (self.num_inputs, 1)))
        return np.reshape(xdot, (self.num_states,))

    def output(self, t, x, u): #, u):
        y = np.matmul(self.C, np.reshape(x, (self.num_states, 1))) # + np.matmul(self.B, np.reshape(x, (self.num_inputs, 1)))
        return np.reshape(y, (self.num_outputs,))

class TransferFunction(StateBlock):

    def __init__(self, num, den, name=None):
        super().__init__(name=name)

        A, B, C, D = tf2ss(num, den)

        self.num_states = A.shape[0]
        num_inputs = B.shape[1]
        num_outputs = C.shape[0]

        for i in range(num_inputs):
            self.add_input('input' + str(i))

        for i in range(num_outputs):
            self.add_output('output' + str(i))

        # set the initial conditions
        self.x0 = np.zeros(self.num_states)

        self.A = np.array(A)
        self.B = np.array(B)
        self.C = np.array(C)
        
        if D is None or np.all( D == 0 ):
            self.is_direct_feedthrough = False
            self.D = None
        else:
            self.is_direct_feedthrough = True
            self.D = D


    def initial(self):
        return self.x0

    # overriding abstract method
    def derivative(self, t, x, u): #, u):
        xdot = np.matmul(self.A, np.reshape(x, (self.num_states, 1))) + np.matmul(self.B, np.reshape(u, (self.num_inputs, 1)))
        return np.reshape(xdot, (self.num_states,))

    def output(self, t, x, u): #, u):
        y = np.matmul(self.C, np.reshape(x, (self.num_states, 1))) # + np.matmul(self.B, np.reshape(x, (self.num_inputs, 1)))
        return np.reshape(y, (self.num_outputs,))

