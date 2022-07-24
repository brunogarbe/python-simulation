from context import *

import matplotlib.pyplot as plt
from simulation import *
from scipy.integrate import solve_ivp
from scipy.signal import tf2ss

num = [1, 3, 3]
den = [1, 2, 1]

A, B, C, D = tf2ss(num, den)
print(A, B, C, D)

# print(A, type(A))

# a = np.matrix('1 2; 3 4')

# a2 = np.matrix([[1, 2], [3, 4]])
# a5 = a3 * a4
# a6 = np.matmul(a3, a4)

# a3t = np.transpose(a3)
# a4t = np.transpose(a4)

# print('a3', a3, type(a3), a3.shape)
# print('a4', a4, type(a4), a4.shape)
# print('a3t', a3t, a3t.shape)
# print('a4t', a4t, a4t.shape)

# a7 = np.matmul(a3, a4)
# a7t = np.matmul(np.transpose(a4), np.transpose(a3))

# print('a7', a7, a7.shape)
# print('a7t', a7t, a7t.shape)

# print(a, type(a))
# #print(a3[0, 0])

# print(a3.shape, len(a3.shape))
# print(a4.shape, len(a4.shape))

# print(a5)
# print(a6)

# z0 = np.zeros(3)

# print(z0)

# b1 = np.array([1, 2, 3])
# b2 = np.array([1])
# b1s = np.reshape(b1, (3, 1))

# print('b1', b1, type(b1), b1.shape)
# print('b1s', b1s, type(b1s), b1s.shape)

# b1ss = np.reshape(b1s, (3,))
# print('b1ss', b1ss, type(b1ss), b1ss.shape)

A = np.array([[0, 1, 0], [0, 0, 1], [-5, -26, -5]])
B = np.array([[0], [0], [1]])
C = np.identity(3)

pulse = Pulse(T=2.5)
step = Step(1.0, 0.5)
ss = StateSpace(A, B, C)
tf = TransferFunction(num, den)

sys = System()
sys.add(pulse)
sys.add(step)
sys.add(ss)
sys.add(tf)
sys.connect(pulse, 0, ss, 0)
sys.connect(pulse, 0, tf, 0)

#ss.derivative(0, b1, b2)

# tf.derivative(0, np.array([1, 1]), np.array([1]))
# print('=====================================================================================')
# tf.output(0, np.array([1, 1]), np.array([1]))

sol = simulate(sys)

fig, axs = plt.subplots(1)
fig.suptitle('Vertically stacked subplots')

major_ticks_top = np.linspace(0, 10, 21)
minor_ticks_top = np.linspace(0, 10, 101)

axs.plot(sol.time, sol['StateSpace.output0'])
axs.plot(sol.time, sol['StateSpace.output1'])
axs.plot(sol.time, sol['StateSpace.output2'])
#axs.plot(sol.time, sol[''])#, marker='o')
#axs.plot(sol.time, sol[''])#, marker='o')
#axs.plot(t_array, y_array[5,:])#, marker='o')

axs.set_xticks(major_ticks_top)
axs.set_xticks(minor_ticks_top,minor=True)
axs.grid(which="major", alpha=0.6)
axs.grid(which="minor", alpha=0.1)

plt.show()

# https://stackoverflow.com/questions/68533913/solving-a-differential-algebraic-equation-dae-problem-with-gekko
# https://stackoverflow.com/questions/23578596/solve-an-implicit-ode-differential-algebraic-equation-dae
# https://stackoverflow.com/questions/68533913/solving-a-differential-algebraic-equation-dae-problem-with-gekko
# https://www.stochasticlifestyle.com/comparison-differential-equation-solver-suites-matlab-r-julia-python-c-fortran/
# https://www.stochasticlifestyle.com/comparison-differential-equation-solver-suites-matlab-r-julia-python-c-fortran/ 