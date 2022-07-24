from context import *
from simulation import *
import matplotlib.pyplot as plt

# System
first_order_gain = 10.0
tau = 10
step = Step(fv=1.0, iv=0.0, ts=0.0)
first = FirstOrder(tau=tau, dc_gain=first_order_gain)

# assert step.num_inputs == 0
# assert step.num_outputs == 1
# assert step.num_states == 0

sys = System()
sys.connect(step, 0, first, 0)

# simulate
sol = simulate(sys)

ref = first_order_gain * (np.ones_like(sol.time) - np.exp(-(1/tau) * sol.time))

fig, axs = plt.subplots(1)
fig.suptitle('Vertically stacked subplots')

major_ticks_top = np.linspace(0, 10, 21)
minor_ticks_top = np.linspace(0, 10, 101)

axs.plot(sol.time, sol['FirstOrder.output']) #, marker='o')
axs.plot(sol.time, ref) #, marker='o')

axs.set_xticks(major_ticks_top)
axs.set_xticks(minor_ticks_top,minor=True)
axs.grid(which="major", alpha=0.6)
axs.grid(which="minor", alpha=0.1)

plt.show()