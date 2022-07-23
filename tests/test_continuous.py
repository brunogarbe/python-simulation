import pytest
from context import *
from simulation import *


@pytest.fixture(params = [1.0, 10.0, 100.0])
def first_order_gain(request):
    return request.param

@pytest.fixture(params = [1.0])
def first_order_tau(request):
    return request.param

def test_first_order_step_response(first_order_gain, first_order_tau):

    # setup the system
    step = Step(fv=1.0, iv=0.0, ts=0.0)
    first = FirstOrder(tau=first_order_tau, dc_gain=first_order_gain)

    # assert step.num_inputs == 0
    # assert step.num_outputs == 1
    # assert step.num_states == 0

    sys = System()
    sys.connect(step, 0, first, 0)

    # simulate
    sol = simulate(sys)

    # generate reference signal based on simulated time
    ref = first_order_gain * (np.ones_like(sol.time) - np.exp(-(1/first_order_tau) * sol.time))

    assert ref == pytest.approx(sol['FirstOrder.output'])


# https://www.tutorialspoint.com/control_systems/control_systems_response_first_order.htm


@pytest.fixture(params = [1.0, 2.0])
def integrator_tau(request):
    return request.param

def test_integrator_feedback_step_response(integrator_tau):

    # setup the system
    step = Step(fv=1.0, iv=0.0, ts=0.0)
    intg = Integrator()
    sum = Sum()

    # assert step.num_inputs == 0
    # assert step.num_outputs == 1
    # assert step.num_states == 0

    sys = System()
    sys.connect(step, 0, sum, 0)
    sys.connect(intg, 0, sum, 1)
    sys.connect(sum, 0, intg, 0)

    # simulate
    sol = simulate(sys)

    # generate reference signal based on simulated time
    first_order_tau = 1.0
    ref = 1.0 * (np.ones_like(sol.time) - np.exp(-(1/first_order_tau) * sol.time))

    assert ref == pytest.approx(sol['Integrator.output'])