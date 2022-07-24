from context import *
from simulation import *
import matplotlib.pyplot as plt

# setup the system
step = Step(fv=1.0, iv=0.0, ts=0.0)
intg = Integrator()
sum = Sum()

sys = System()
sys.connect(step, 0, sum, 0)
sys.connect(intg, 0, sum, 1)
sys.connect(sum, 0, intg, 0)

#sys.prepare()

#print(sys)

# simulate
sol = simulate(sys)

# reference
first_order_tau = 1.0
ref = 1.0 * (np.ones_like(sol.time) - np.exp(-(1/first_order_tau) * sol.time))

# plot the results
fig, axs = plt.subplots(1)
fig.suptitle('Vertically stacked subplots')

major_ticks_top = np.linspace(0, 10, 21)
minor_ticks_top = np.linspace(0, 10, 101)

axs.plot(sol.time, sol['Integrator.output']) #, marker='o')
axs.plot(sol.time, sol['Sum.output']) #, marker='o')
axs.plot(sol.time, sol['Step.output']) #, marker='o')
axs.plot(sol.time, ref) #, marker='o')

axs.set_xticks(major_ticks_top)
axs.set_xticks(minor_ticks_top,minor=True)
axs.grid(which="major", alpha=0.6)
axs.grid(which="minor", alpha=0.1)

plt.show()