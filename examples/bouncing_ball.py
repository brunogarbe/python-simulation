from context import *

import matplotlib.pyplot as plt
from simulation import *

class BouncingBall(StateBlock):

    class FSM(Enum):
        MOVING = 1
        STOP = 2

    def __init__(self, x0=0.0, v0=0.0, name=None):
        super().__init__(name=name)

        self.num_states = 2
        
        self.x0 = np.array([x0, v0])
        self.fsm = self.FSM.MOVING

        self.add_output('height')
        self.add_output('velocity')

        self.add_event(self.event_hitground, self.handler_hitground)

    #    return
    def initial(self):
        return self.x0

    # overriding abstract method
    def derivative(self, t, x, u):
        if self.fsm == self.FSM.MOVING:
            dxdt_0 = x[1]
            dxdt_1 = -9.80665
        else:
            dxdt_0 = 0
            dxdt_1 = 0
        return np.array([dxdt_0, dxdt_1])

    def output(self, t, x, u):
        return x

    # Event Hit the Ground
    def event_hitground(self, t, x):
        return -x[0]

    def handler_hitground(self, t, x):
        if abs(x[1]) < 0.1:
            x[0] = 0.0
            x[1] = 0.0
            self.fsm = self.FSM.STOP
        else:
            x[1] = -0.9*x[1]
        return x


# https://mbe.modelica.university/behavior/discrete/bouncing/
# model BouncingBall "The 'classic' bouncing ball model"
#   type Height=Real(unit="m");
#   type Velocity=Real(unit="m/s");
#   parameter Real e=0.8 "Coefficient of restitution";
#   parameter Height h0=1.0 "Initial height";
#   Height h "Height";
#   Velocity v(start=0.0, fixed=true) "Velocity";
# initial equation
#   h = h0;
# equation
#   v = der(h);
#   der(v) = -9.81;
#   when h<0 then
#     reinit(v, -e*pre(v));
#   end when;
# end BouncingBall;

ball = BouncingBall(5.0)

sys = System()
sys.add(ball)
#sys.add_event(ball, ball.HitGroundEvent, ball.test_handler)

sys.prepare()

res = simulate(sys)
t_array = res.time

fig, axs = plt.subplots(2)
fig.suptitle('Vertically stacked subplots')

major_ticks_top = np.linspace(0,10,21)
minor_ticks_top = np.linspace(0,10,101)

axs[0].plot(t_array, res['BouncingBall.height'])#, marker='o') 
axs[0].set_xticks(major_ticks_top)
#axs[0].set_yticks(major_ticks_top)
axs[0].set_xticks(minor_ticks_top,minor=True)
axs[0].grid(which="major", alpha=0.6)
axs[0].grid(which="minor", alpha=0.1)

axs[1].plot(t_array, res['BouncingBall.velocity'])#, marker='o')
axs[1].set_xticks(major_ticks_top)
axs[1].set_xticks(minor_ticks_top,minor=True)
axs[1].grid(which="major", alpha=0.6)
axs[1].grid(which="minor", alpha=0.1)

plt.show()

