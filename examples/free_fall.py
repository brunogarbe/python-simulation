from context import *

import matplotlib.pyplot as plt
from simulation import *

class FreeFall(StateBlock):

    def __init__(self, x0=0.0, v0=0.0, name=None):
        super().__init__(name=name)

        self.num_states = 2
        
        self.x0 = np.array([x0, v0])

        self.add_output('height')
        self.add_output('velocity')

    def initial(self):
        return self.x0

    # overriding abstract method
    def derivative(self, t, x, u):
        dxdt_0 = x[1]
        dxdt_1 = -9.80665
        return np.array([dxdt_0, dxdt_1])

    def output(self, t, x, u):
        return x

    # Event Hit the Ground
    def event_hitground(self, t, x, u):
        return -x[0]

sys = System()

fall = FreeFall(10.0)
fall.add_event(fall.event_hitground, sys.exit)
sys.add(fall)


res = simulate(sys)
t_array = res.time

fig, axs = plt.subplots(2)
fig.suptitle('Free fall example')

major_ticks_top = np.linspace(0, 1.5, 11)
minor_ticks_top = np.linspace(0, 1.5, 51)

axs[0].plot(t_array, res['FreeFall.height'])#, marker='o')
axs[0].plot(t_array[-1], res['FreeFall.height'][-1], marker='o')
axs[0].text(t_array[-1], res['FreeFall.height'][-1], 't = ' + str(t_array[-1]))
axs[0].set_xticks(major_ticks_top)
#axs[0].set_yticks(major_ticks_top)
axs[0].set_xticks(minor_ticks_top,minor=True)
axs[0].grid(which="major", alpha=0.6)
axs[0].grid(which="minor", alpha=0.1)

axs[1].plot(t_array, res['FreeFall.velocity'])#, marker='o')
axs[1].plot(t_array[-1], res['FreeFall.velocity'][-1], marker='o')
axs[1].text(t_array[-1], res['FreeFall.velocity'][-1], 'v = ' + str(res['FreeFall.velocity'][-1]))
axs[1].set_xticks(major_ticks_top)
axs[1].set_xticks(minor_ticks_top,minor=True)
axs[1].grid(which="major", alpha=0.6)
axs[1].grid(which="minor", alpha=0.1)

plt.show()

