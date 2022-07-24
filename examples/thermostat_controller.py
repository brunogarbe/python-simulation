from context import *

import matplotlib.pyplot as plt
from simulation import *
from scipy.integrate import solve_ivp


class Thermostat(BasicBlock):

    def __init__(self, name=None):
        super().__init__(name=name)

        self.fsm = True
        self.is_direct_feedthrough = True

        self.add_input('input')
        self.add_output('output')
        
        self.add_event(self.event_change, self.handler_change)

    def output(self, t, x, u):
        if self.fsm == True:
            return np.array([30.0])
        else:
            return np.array([0.0])

    # Event Hit the Ground
    def event_change(self, t, x, u):
        if self.fsm == True: 
            return u[0] - 21
        else:
            return 19 - u[0]
            
    def handler_change(self, t, x):
        if self.fsm == True:
            self.fsm = False
        else:
            self.fsm = True
        return x

class Thermostat2(BasicBlock):

    def __init__(self, name=None):
        super().__init__(name=name)

        self.fsm = True
        self.is_direct_feedthrough = True

        self.add_input('input')
        self.add_output('output')

    def output(self, t, x, u):
        if self.fsm == True:
            if u[0] > 21:
                self.fsm = False
                return np.array([0.0])
            else:
                return np.array([30.0])
        else:
            if u[0] < 19:
                self.fsm = True
                return np.array([30.0])
            else:
                return np.array([0.0])


class Plant(StateBlock):

    def __init__(self, x0=0.0, tau=1.0, dc_gain=1.0, name=None):
        super().__init__(name=name)

        self.num_states = 1
        self.fsm = True

        #self.add_input('input')
        self.add_output('output')

        # set the initial conditions
        self.x0 = np.array([x0])

        # set the parameters
        self.tau = tau
        self.K = dc_gain

        # TODO: check for zeros in the parameters!!!
        self.add_event(self.event_change, self.handler_change)

    def initial(self):
        return self.x0

    # overriding abstract method
    def derivative(self, t, x, u): #, u):
        if self.fsm == True:
            u = 30.0
        else:
            u = 0.0
        dxdt_0 = -(1.0/self.tau)*x[0] + (self.K/self.tau)*u

        return np.array([dxdt_0])

    def output(self, t, x, u): #, u):
        return x

    # Event Hit the Ground
    def event_change(self, t, x, u):
        #print('!', x)
        if self.fsm == True: 
            return x[0] - 21
        else:
            return 19 - x[0]
            
    def handler_change(self, t, x):
        if self.fsm == True:
            self.fsm = False
        else:
            self.fsm = True
        return x

        
sys = System()

thermo = Thermostat2()
first = FirstOrder()
plant = Plant()
thermo2 = Thermostat()
first2 = FirstOrder()

sys.add(plant)

sys.add(first)
sys.add(thermo)

sys.add(first2)
sys.add(thermo2)

sys.connect(thermo, 0, first, 0)
sys.connect(first, 0, thermo, 0)

sys.connect(thermo2, 0, first2, 0)
sys.connect(first2, 0, thermo2, 0)


sol = simulate(sys)

fig, axs = plt.subplots(1)
fig.suptitle('Vertically stacked subplots')

major_ticks_top = np.linspace(0, 10, 21)
minor_ticks_top = np.linspace(0, 10, 101)

#axs.plot(t_array, res['Step.output'])#, marker='o')
axs.plot(sol.time, sol['Thermostat.output'])#, marker='o')
#axs.plot(sol.time, sol[''])#, marker='o')
#axs.plot(sol.time, sol[''])#, marker='o')
axs.set_xticks(major_ticks_top)
axs.set_xticks(minor_ticks_top, minor=True)
axs.grid(which="major", alpha=0.6)
axs.grid(which="minor", alpha=0.1)


plt.show()

# https://stackoverflow.com/questions/68533913/solving-a-differential-algebraic-equation-dae-problem-with-gekko
# https://stackoverflow.com/questions/23578596/solve-an-implicit-ode-differential-algebraic-equation-dae
# https://stackoverflow.com/questions/68533913/solving-a-differential-algebraic-equation-dae-problem-with-gekko
# https://www.stochasticlifestyle.com/comparison-differential-equation-solver-suites-matlab-r-julia-python-c-fortran/
# https://www.stochasticlifestyle.com/comparison-differential-equation-solver-suites-matlab-r-julia-python-c-fortran/ 