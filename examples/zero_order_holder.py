from context import *

from simulation import *

import matplotlib.pyplot as plt

# System
sin_in = Sine(ws=1)
zoh = ZeroOrderHold()
disc = DiscreteTeste()

sys = System()
sys.connect(sin_in, 0, zoh, 0)
sys.connect(sin_in, 0, disc, 0)

res = simulate(sys)

fig, axs = plt.subplots(1)
fig.suptitle('Vertically stacked subplots')

major_ticks_top = np.linspace(0, 10, 21)
minor_ticks_top = np.linspace(0, 10, 101)

axs.plot(res.time, res['Sine.output']) #, marker='o')
axs.plot(res.time, res['ZeroOrderHold.output']) #, marker='o')
#axs.plot(res.time, res['FirstOrder.output']) #, marker='o')
axs.plot(res.time, res['DiscreteTeste.output']) #, marker='o')

axs.set_xticks(major_ticks_top)
axs.set_xticks(minor_ticks_top,minor=True)
axs.grid(which="major", alpha=0.6)
axs.grid(which="minor", alpha=0.1)

plt.show()