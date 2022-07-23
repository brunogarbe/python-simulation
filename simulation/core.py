from abc import ABC, abstractmethod
from enum import Enum
from inspect import signature
from scipy.optimize import fsolve
import numpy as np
import networkx as nx
from scipy.integrate import solve_ivp

from .steppers import stepper

# https://www.youtube.com/watch?v=M3wVGZrP0ZM&t=152s
# https://jmodelica.org/assimulo/
# https://pypi.org/project/Assimulo/
# https://github.com/modelon-community/Assimulo
# https://pypi.org/project/python-sundials/#:~:text=python%2Dsundials%20is%20a%20Cython,Jon%20Olav%20Vik.&text=The%20CVODE%20and%20IDA%20solvers,exception%20on%20finding%20a%20root.
# https://lcvmwww.epfl.ch/software/daepy/
# https://github.com/alastairflynn/daepy

# https://stackoverflow.com/questions/1051254/check-if-python-package-is-installed


# https://stackoverflow.com/questions/60136367/simultaneous-event-detection-using-solve-ivp
def gen_event(call, idx, sz, sys, input_idx):
    #print('$', call, idx, sz, input_idx)
    
    def event(t, x):
        y = sys.output(t, x)
        #print('input_idx', input_idx)
        u = y[input_idx] if len(input_idx) != 0 else None
        #print('&', y, y[idx:idx+sz], idx, sz)
        return call(t, x[idx:idx+sz], u)
    event.terminal = True
    event.direction = 1
    return event
        
class Channel:
    def __init__(self, block=None, port=None):
        self.block = block
        self.port = port
    
    def __str__(self):
        return f'{self.block}|{self.port}'

class InputChannel:
    def __init__(self, name, unit=None, block=None, port=None):
        self.name = name
        self.unit = unit
        self.block = block
        self.port = port
    
    #def __str__(self):
    #    return f'{self.name}|{self.block.name}|{self.port}'
        
class OutputChannel:
    def __init__(self, name, unit=None, block=None, port=None):
        self.name = name
        self.unit = unit
        self.channels = []
    
    def __str__(self):
        return f'{self.name}|{self.channels}'

class EventHandler:
    def __init__(self, block, event, handler):
        self.block = block
        self.event = event
        self.handler = handler

class BasicBlock(ABC):

    # static counter
    instance_counter = 0

    # Constructor
    def __init__(self, name=None) -> None:
        super().__init__()
        BasicBlock.instance_counter += 1
        self.id = BasicBlock.instance_counter
        self.name = type(self).__name__ if name is None else name
        
        self.num_outputs = 0
        self.num_states = 0
        self.num_inputs = 0
        self.is_direct_feedthrough = False

        self.inputs = []
        self.outputs = []
        self.states = []
        self.events = []

        self.input_index = None
        self.output_index = None
        self.state_index = None

    @abstractmethod
    def output(self, t, x, u):
        pass

    def add_event(self, ev, handler=None):
        self.events.append(EventHandler(self, ev, handler))
    
    def add_input(self, name=None, unit=None):
        name = name if name is not None else 'input_' + str(self.num_inputs)
        self.inputs.append(InputChannel(name))
        self.num_inputs += 1

    def add_output(self, name=None, unit=None):
        name = name if name is not None else 'output_' + str(self.num_outputs)
        self.outputs.append(OutputChannel(name))
        self.num_outputs += 1

    def set_input(self, input_port, input_name, input_unit=None):
        self.inputs[input_port].name = input_name
        self.inputs[input_port].unit = input_unit
    
    def set_output(self, output_port, output_name, output_unit=None):
        self.outputs[output_port].name = output_name
        self.outputs[output_port].unit = output_unit

    def set_input_connection(self, input_port, block, output_port):
        self.inputs[input_port].block = block
        self.inputs[input_port].port = output_port
    
    def add_output_connection(self, output_port, block, input_port):
        self.outputs[output_port].channels.append(Channel(block, input_port))
    
    def __str__(self):
        ret = self.name
        ret += '['
        ret += str(self.id)
        ret += '] '
        
        # if self.num_inputs > 0:
        #     ret += str(self.num_inputs)
        #     ret += 'i{'
        #     for i in range(self.num_inputs):
        #         if i in self.inputs:
        #             if 'block' in self.inputs[i]:
        #                 ret += self.inputs[i]['block'].name
        #                 if 'name' in self.inputs[i]['block'].outputs[self.inputs[i]['out_port']]:
        #                     ret += '.'
        #                     ret += self.inputs[i]['block'].outputs[self.inputs[i]['out_port']]['name']
        #                 ret += ':'
        #                 ret += str(self.inputs[i]['out_port'])
        #                 ret += '->'
        #             if 'name' in self.inputs[i]:
        #                 ret += self.inputs[i]['name']
        #                 ret += ':'
        #         ret += str(i)
                    
        #     ret += '}'
                    
        # if self.num_outputs > 0:
        #     ret += ' '
        #     ret += str(self.num_outputs)
        #     ret += 'o{'
        #     for i in range(self.num_outputs):
        #         if i in self.outputs:
        #             if 'name' in self.outputs[i]:
        #                 ret += self.outputs[i]['name']
        #                 ret += ':'
        #         ret += str(i)

        #         if 'block' in self.outputs[i]:
        #             ret += '->'
        #             ret += self.outputs[i]['block'].name
        #             if 'name' in self.outputs[i]['block'].inputs[self.outputs[i]['in_port']]:
        #                 ret += '.'
        #                 ret += self.outputs[i]['block'].inputs[self.outputs[i]['in_port']]['name']
        #             ret += ':'
        #             ret += str(self.outputs[i]['in_port'])

        #         if i != self.num_outputs - 1:
        #             ret += ' '
        #     ret += '}'

        # ret += ' '
        # ret += str(self.input_index)

        # if self.num_states > 0:
        #     ret += ' '
        #     ret += str(self.num_states)
        #     ret += 's{'
        #     for i in range(self.num_states):
        #         if i in self.states:
        #             if 'name' in self.states[i]:
        #                 ret += self.states[i]['name']
        #                 ret += ':'
        #         ret += str(i)
        #     ret += '}'
                   
        return ret


class StateBlock(BasicBlock):

    # Constructor
    def __init__(self, name=None) -> None:
        super().__init__(name=name)

    # Continuous obligatory methods are initial and derivative
    @abstractmethod
    def initial(self):
        pass

    @abstractmethod
    def derivative(self, t, x, u):
        pass

class Subsystem:
    def __init__(self):
        self.blocks = []
        self.connections = []
        self.inputs = []
        self.outputs = []
        #pass
    
    def add(self, bl):
        self.blocks.append(bl)

    def connect(self, bl_out, out_port, bl_in, in_port):

        # if bl_in.inputs[in_port].block is not None and bl_in.inputs[in_port].port is not None:
        #     raise Exception("Channel already used!!")
            
        # bl_out.add_output_connection(out_port, bl_in, in_port)
        # bl_in.set_input_connection(in_port, bl_out, out_port)

        self.connections.append( (bl_out, out_port, bl_in, in_port) )

        # if bl_in.is_direct_feedthrough == True:
        #     self.direct_connections.append( (bl_out.id, bl_in.id) )

    def add_input(self, bl, port):
        self.inputs.append(Channel(bl, port))

    def add_output(self, bl, port):
        self.outputs.append(Channel(bl, port))

    def get_input(self, port):
        return self.inputs[port]

    def get_output(self, port):
        return self.outputs[port]

        
class System(StateBlock):
    def __init__(self):
        super().__init__()

        self.blocks = []
        self.all_events = []
        self.sys_order = []

        self.connections = []
        self.direct_connections = []

        self.outnames = []
        
        self.exit_flag = False

    def add(self, bl):

        if isinstance(bl, Subsystem) == True:
            print('BINGO!!!')
            for b in bl.blocks:
                print('bingo')
                self.blocks.append(b)

            # connect the internal
            for bl_out, out_port, bl_in, in_port in bl.connections:
                print('con')
                self.connect(bl_out, out_port, bl_in, in_port)

                # TODO: check for events!!

        else:
            self.blocks.append(bl)

            for ev in bl.events:
                self.__aux_add_event(ev.block, ev.event, ev.handler)
        
    def __aux_add_event(self, model, event, handler=None):
        self.all_events.append( [model, event, handler, 0, 0] )

    def connect(self, bl_out, out_port, bl_in, in_port):

        if isinstance(bl_out, Subsystem) == True:
            channel = bl_out.get_output(out_port)
            bl_out = channel.block
            out_port = channel.port

        if isinstance(bl_in, Subsystem) == True:
            channel = bl_in.get_input(in_port)
            bl_in = channel.block
            in_port = channel.port

        # check blocks are in self.blocks, and if not add them
        if bl_out not in self.blocks:
            #print('BINGO!!', bl_out.name)
            self.add(bl_out)

        if bl_in not in self.blocks:
            #print('BINGO!!', bl_in.name)
            self.add(bl_in)

        
        # check that the connections are not already used
        if bl_in.inputs[in_port].block is not None and bl_in.inputs[in_port].port is not None:
            raise Exception("Channel already used!!")
            
        bl_out.add_output_connection(out_port, bl_in, in_port)
        bl_in.set_input_connection(in_port, bl_out, out_port)

        print('connect', bl_out.name, out_port, bl_in.name, in_port)

        self.connections.append( (bl_out, out_port, bl_in, in_port) )

        if bl_in.is_direct_feedthrough == True:
            self.direct_connections.append( (bl_out.id, bl_in.id) )

    def prepare(self):

        # TODO: CHECK FOR THE NUMBER OF len(inputs) = num_inputs

        # creates the graph for the system
        graph = nx.DiGraph()

        # add the nodes to the graph and creates a dictionary based on the blocks id's
        block_id_dict = {}
        for bl in self.blocks:
            graph.add_node(bl.id)
            block_id_dict[bl.id] = bl

        # add the edges
        for con in self.direct_connections:
            graph.add_edge(con[0], con[1])

        # get the order of the blocks by topological sort
        block_order = list(nx.topological_sort(graph))

        print(block_order)

        # calculate the state/output indexes
        self.num_states = 0
        self.num_outputs = 0
        idx_state = 0
        idx_output = 0
        self.sys_order = []
        state_dict = {}
        output_dict = {}

        for bl_id in block_order:
            sys = block_id_dict[bl_id]
            self.num_states += sys.num_states
            self.num_outputs += sys.num_outputs
            for i in range(sys.num_outputs):
                self.outnames.append(sys.name + '.' + sys.outputs[i].name)
            self.sys_order.append( (sys, idx_state, sys.num_states, idx_output, sys.num_outputs) )
            state_dict[sys] = (idx_state, sys.num_states)
            output_dict[sys] = (idx_output, sys.num_outputs)

            sys.state_index = (idx_state, sys.num_states)
            sys.output_index = (idx_output, sys.num_outputs)

            idx_state += sys.num_states
            idx_output += sys.num_outputs

        print('idx_state', idx_state)
        print('num_states', self.num_states)
        print('num_outputs', self.num_outputs)

        print(self.sys_order)

        # calculate the input indexes
        for bl_id in block_order:
            sys = block_id_dict[bl_id]
            input_idx = []
            for in_port in range(sys.num_inputs):
                idx_in_out_matrix = output_dict[sys.inputs[in_port].block][0] + sys.inputs[in_port].port
                input_idx.append(idx_in_out_matrix)
            sys.input_index = np.array(input_idx) 

        # model, ev, handler, idx, sz in
        for i, _ in enumerate(self.all_events):
            model = self.all_events[i][0]
            idx_state, size_state = state_dict[model]
            self.all_events[i][3] = idx_state
            self.all_events[i][4] = size_state

        self.out = np.zeros(self.num_outputs)

    def initial(self):
        x0 = np.zeros(self.num_states)
        for sys, idx_states, num_states, _, _ in self.sys_order:
            if sys.num_states > 0:
                x0[idx_states:idx_states+num_states] = sys.initial()

        return x0
        
    def derivative(self, t, x):
        dx = np.zeros(self.num_states)
        out = self.output(t, x)
        
        for bl, idx_state, num_states, _, _ in self.sys_order:
            # if sys.num_inputs > 0:
            #     u = out[sys.input_index]
            # else:
            #     u = None
            u = out[bl.input_index] if bl.num_inputs > 0 else None

            if num_states != 0:
                dx[idx_state:idx_state+num_states] = bl.derivative(t, x[idx_state:idx_state+num_states], u)

        return dx

    def output(self, t, x):
        out = np.zeros(self.num_outputs)
        
        for bl, idx_state, num_states, idx_out, num_outputs in self.sys_order:
            # if sys.num_inputs > 0:
            #     u = out[sys.input_index]
            # else:
            #     u = None
            
            if bl.is_direct_feedthrough == False:
                u = None
            else:
                u = out[bl.input_index] if (bl.num_inputs > 0) and (bl.is_direct_feedthrough) else None
            
            out[idx_out:idx_out+num_outputs] = bl.output(t, x[idx_state:idx_state+num_states], u)
        return out
        
    def exit(self):
        self.exit_flag = True

    def get_events_list(self):
        ret = []
        for model, ev, _, idx, sz in self.all_events:
            ret.append(gen_event(ev, idx, sz, self, model.input_index))
        return ret
        
    def get_event_indexes(self, idx):
        idx_state = self.all_events[idx][3]
        size_state = self.all_events[idx][4]
        return idx_state, size_state
        
    def get_event_handler(self, idx):
        handler = self.all_events[idx][2]
        return handler

    def __str__(self):
        ret = '#####################################################################################\n'
        
        ret += '# Blocks:\n'
        for bl in self.blocks:
            ret += f'{"":3} {bl.name} {bl.id}\n'
        
        ret += '# Connections:\n'
        for bl, idx, num_states, idx_out, num_outputs in self.sys_order:
            for i in range(bl.num_outputs):
                dir_feed = '*' if bl.is_direct_feedthrough == True else ''
                full_name_out = dir_feed + bl.name + '.' + bl.outputs[i].name + '<' + str(i) + '>'
                
                if len(bl.outputs[i].channels) == 0:
                    ret += f'{idx_out+i:3} {full_name_out:20} -<\n'
                    
                elif len(bl.outputs[i].channels) == 1:
                    bl_in = bl.outputs[i].channels[0].block
                    in_port = bl.outputs[i].channels[0].port
                    dir_feed = '*' if bl_in.is_direct_feedthrough == True else ''
                    full_name_in = dir_feed + bl_in.name + '.' + bl_in.inputs[in_port].name + '<' + str(in_port) + '>'
                    ret += f'{idx_out+i:3} {full_name_out:20} -> {full_name_in:20} {bl_in.input_index}\n'
                
                else:
                    for j, channel in enumerate(bl.outputs[i].channels):
                        bl_in = bl.outputs[i].channels[j].block
                        in_port = bl.outputs[i].channels[j].port
                        dir_feed = '*' if bl_in.is_direct_feedthrough == True else ''
                        full_name_in = dir_feed + bl_in.name + '.' + bl_in.inputs[in_port].name + '<' + str(in_port) + '>'
                        if j == 0:
                            ret += f'{idx_out+i:3} {full_name_out:20} -> {full_name_in:20} {bl_in.input_index}\n'
                        else:
                            ret += f'{"":3} {"":20} -> {full_name_in:20} {bl_in.input_index}\n'
                            
        ret += '# Events:\n'
        for i, _ in enumerate(self.all_events):
            model = self.all_events[i][0]
            #idx_state, size_state = state_dict[model]
            event = self.all_events[i][1]
            handler = self.all_events[i][2]
            idx_state = self.all_events[i][3]
            size_state = self.all_events[i][4]
            
            event_name = event.__self__.name + '.' + event.__name__
            handler_name = handler.__self__.name + '.' + handler.__name__
            ret += f'{i:3} {model.name:20} {event_name:30} {handler_name:30} [{idx_state}:{idx_state+size_state}]\n'
            #self.all_events.append( [model, event, handler, 0, 0] )
        
        return ret


class Scheduler:
    def __init__(self):
        self.events_table = []
    
    def add(self, time, handler):
        
        for i, (ti, _) in enumerate(self.events_table):
            if abs(time - ti) <= 0.00001:
                self.events_table[i][1].append(handler)
                return
        
        # if no event on time
        self.events_table.append([time, [handler]])
        self.events_table.sort(key=lambda tup: tup[0])  # sorts in place
        
    def pop(self):
        self.events_table.pop(0)
    
    def next_time(self):
        return self.events_table[0][0]

    def call_handlers(self, X, Y):
        #print('t_event =', self.events_table[0][0], X, Y)
        for handler in self.events_table[0][1]:
            if handler is not None:
                #print()
                #print('handler', handler, handler.__self__.state_index, handler.__self__.output_index, handler.__self__.input_index)
                x = None if handler.__self__.num_states == 0 else X[idx_start:idx_end]
                u = None if handler.__self__.num_inputs == 0 else Y[handler.__self__.input_index]
                idx_start = handler.__self__.state_index[0]
                idx_end = handler.__self__.state_index[0] + handler.__self__.state_index[1]
                print(Y, handler.__self__.input_index)
                handler(x, u)
                
    def has_ended(self):
        return len(self.events_table) == 0 
    


class Results:
    def __init__(self, time=None, outputs=None, names=None):
        self.time = time
        self.outputs = outputs
        self.outnames = names

    def __getitem__(self, item):
        idx = self.outnames.index(item)
        return self.outputs[idx,:]

####################################################################################################

def simulate(sys, t_final=10.0, dt_max = 0.01):

    #print('oi')
    sys.prepare()

    print('=======================================================================================')
    print('num_states', sys.num_states)

    if sys.num_states == 0:
        return simulate_nostates(sys)
    else:

        return simulate_scipy(sys)


def simulate_nostates(sys):

    t_start = 0.0
    t_final=10.0
    dt_max = 0.1
    X_null = np.array([])

    # start the scheduler
    scheduler = Scheduler()
    scheduler.add(t_final, None)

    for bl in sys.sys_order:
        if hasattr(bl[0], 'add_schedule') == True:
            bl[0].add_schedule(scheduler, 0.0, t_final)

    t_array = np.empty(0)
    x_array = np.empty((sys.num_states, 0))
    y_array = np.empty((sys.num_outputs, 0))
    events_array = []

    # exit_flag = False

    # auxiliar funtion for test conditions
    # def test_condition(func, condition, dt, t, X_cur, idx, sz):
    #     X_new = stepper(func, dt, t, X_cur)
    #     return condition(t, X_new[idx:idx+sz])

    # t = 0
    # idx = 0



    # TODO: check for events that happens before t_start

    t_interval_end = t_start
    while scheduler.has_ended() == False:
        #print('==========================================================================================')
        t_interval_start = t_interval_end
        t_interval_end = scheduler.next_time()
        #print(f'({t_interval_start}, {t_interval_end})')

        for t_cur in np.arange(t_interval_start, t_interval_end, dt_max):
            Y_cur = sys.output(t_cur, X_null)

            #print(t_cur, Y_cur)
            # TODO: check for events!!!
            
            t_array = np.append(t_array, t_cur)
            y_array = np.append(y_array, Y_cur.reshape((sys.num_outputs,1)), axis=1)
        #print(samples)

        # Y_try = sys.output(t_interval_start, X_null)
        # print(Y_try)

        # #X_initial = sol.y[:,-1]
        # #print('X_initial', X_initial)
        # #print('y_array[:,-1]', y_array[:,-1])
        scheduler.call_handlers(X_null, np.array([0, 0, 0, 0, 0, 0, 0]))
        scheduler.pop()

    res = Results(time=t_array, outputs=y_array, names=sys.outnames)

    return res

def simulate_scipy(sys):
    sys.prepare()

    #def simulate2(sys, t_final=10.0):
    t_start = 0.0
    t_final=10.0


    all_events = sys.get_events_list() #[event03]

    #print(sys)

    X_cur = sys.initial()

    # start the scheduler
    scheduler = Scheduler()
    scheduler.add(t_final, None)

    for bl in sys.sys_order:
        if hasattr(bl[0], 'add_schedule') == True:
            bl[0].add_schedule(scheduler, 0.0, t_final)
            
    #print(scheduler.events_table)

    t_array = np.empty(0)
    x_array = np.empty((sys.num_states, 0))
    y_array = np.empty((sys.num_outputs, 0))
    events_array = []

    #print('sys.num_states', sys.num_states)
    #print('x_array', x_array)

    X_initial = sys.initial()
    t_interval_end = t_start

    #print('X_initial', X_initial)

    while scheduler.has_ended() == False:
        #print('==========================================================================================')
        t_interval_start = t_interval_end
        t_interval_end = scheduler.next_time()
        
        #print('t_interval', t_interval_start, t_interval_end)
        #print('X_initial', X_initial)
        
        while True:
        
            sol = solve_ivp(sys.derivative, (t_interval_start, t_interval_end), X_initial, events=all_events, max_step=0.01)
            
            if sol.status == 1:
                #print('Success, termination event occured.')
            
                t = sol.t_events[0]
                x = sol.y_events[0][0,:]
                
                t_interval_start = sol.t_events[0]
                X_initial = x #sol.y_events[0].reshape((sys.num_states,1))
                
                for i, event_function in enumerate(all_events):
                    idx, sz = sys.get_event_indexes(i)
                    handler = sys.get_event_handler(i)
                    #print('oi')
                    value = event_function(t, X_initial)
                    
                    #print(f'% {i} {value} [{idx}:{sz}] {handler}')
                    if abs(value) < 0.00001:
                        #print('bingo')
                        X_initial[idx:sz] = handler(t, X_initial[idx:sz])
                
                #print('@', t, x, X_initial)
                
                sz = sol.y.size // len(sol.y)
                y_sol = np.empty((sys.num_outputs, sz))
                for i in range(sz):
                    y_sol[:,i] = sys.output(sol.t[i], sol.y[:,i])

                t_array = np.append(t_array, sol.t)
                x_array = np.append(x_array, sol.y, axis=1)
                y_array = np.append(y_array, y_sol, axis=1)
            
            if sol.status == 0:
                #print('Success, t end reached.')

                #print(sol)
            
                sz = sol.y.size // len(sol.y)
                y_sol = np.empty((sys.num_outputs, sz))
                for i in range(sz):
                    y_sol[:,i] = sys.output(sol.t[i], sol.y[:,i])
                
                t_array = np.append(t_array, sol.t)
                x_array = np.append(x_array, sol.y, axis=1)
                y_array = np.append(y_array, y_sol, axis=1)
                
                break
        
        X_initial = sol.y[:,-1]
        print('X_initial', X_initial)
        print('y_array[:,-1]', y_array[:,-1])
        scheduler.call_handlers(X_initial, y_array[:,-1])
        scheduler.pop()

        if sys.exit_flag is True:
            break

    res = Results(time=t_array, outputs=y_array, names=sys.outnames)
        
    return res

def simulate_old(sys, t_final=10.0, dt_max = 0.01):

    sys.prepare()
    
    X_cur = sys.initial()

    # start the scheduler
    scheduler = Scheduler()
    scheduler.add(t_final, None)

    for bl in sys.sys_order:
        if hasattr(bl[0], 'add_schedule') == True:
            bl[0].add_schedule(scheduler, 0.0, t_final)
    
    t_array = np.empty(0)
    x_array = np.empty((sys.num_states, 0))
    y_array = np.empty((sys.num_outputs, 0))
    events_array = []

    exit_flag = False

    # auxiliar funtion for test conditions
    def test_condition(func, condition, dt, t, X_cur, idx, sz):
        X_new = stepper(func, dt, t, X_cur)
        return condition(t, X_new[idx:idx+sz])

    t = 0
    idx = 0
    
    while True:
        
        Y_cur = sys.output(t, X_cur)

        # get the outputs in initial t0
        # Output the data
        t_array = np.append(t_array, t)
        x_array = np.append(x_array, X_cur.reshape((sys.num_states,1)), axis=1)
        y_array = np.append(y_array, Y_cur.reshape((sys.num_outputs,1)), axis=1)
        
        # TODO must check for exit here!!!
        #if exit_flag is True:
        if sys.exit_flag is True:
            #print('EXIT FLAG', t)
            break
        
        if abs(t - t_final) <= 0.0001:
            #print('t_final', t)
            break
        
        # calculate the next state using the dt_max
        # TODO: check for simulation final time
        t_try = t + dt_max
        dt = dt_max
        scheduler_guard = False
        
        if abs(t_try - scheduler.next_time()) <= 0.0001:
            # print('CALL HANDLERS')
            # scheduler.call_handlers()
            # scheduler.pop()
            scheduler_guard = True
        elif t_try > scheduler.next_time():
            t_try = scheduler.next_time()
            dt = scheduler.next_time() - t
            scheduler_guard = True
        
        X_try = stepper(sys.derivative, dt, t_try, X_cur)
    
        all_conditions = []
        all_handlers = []
        all_index = []
        for model, ev, handler, idx, sz in sys.all_events:
            all_conditions.append(ev(t_try, X_try[idx:idx+sz]))
            all_handlers.append(handler)
            all_index.append( (idx, sz) )
            
        
        if any(c > 0 for c in all_conditions):

            dt_min = dt_max
            event_handler_min = None
            idx_min = 0
            sz_min = 0
            model_min = None
            for i, c in enumerate(all_conditions):
                if c > 0:
                    model = sys.all_events[i][0]
                    evnt = sys.all_events[i][1]
                    idx, sz = all_index[i]
                    dte_temp = fsolve(lambda dt, t, X_cur, idz, sz: test_condition(sys.derivative, evnt, dt, t, X_cur, idx, sz), 0.5*dt, args=(t, X_cur, idx, sz))
                    dte = dte_temp[0]

                    if dte < dt_min:
                        dt_min = dte
                        event_handler_min = all_handlers[i]
                        idx_min = idx
                        sz_min = sz
                        model_min = model
            
            # calculate the state for this dt_min
            X_new = stepper(sys.derivative, dt_min, t, X_cur)
            X_cur = X_new
            t += dt_min

            events_array.append( (t, model_min.name, event_handler_min.__name__))

            if event_handler_min is not None:
                sig = signature(event_handler_min)

                if len(sig.parameters) == 0:
                    event_handler_min()
                elif len(sig.parameters) == 2:
                    X_cur_partial = event_handler_min(t, X_cur[idx_min:idx_min+sz_min])
                    X_cur[idx_min:idx_min+sz_min] = X_cur_partial
                else:
                    raise Exception("Date provided can't be in the past")

            continue

        else: # no condition was met
        
            X_cur = X_try
            t = t_try

        if scheduler_guard == True:
            scheduler.call_handlers()
            scheduler.pop()
        
        idx += 1

    res = Results(time=t_array, outputs=y_array, names=sys.outnames)

    return res

    
# # References
# https://www.mathworks.com/help/simulink/sfg/s-function-concepts.html
# https://pypi.org/project/Events/
# https://www.magnoliasci.com/helpdocs/DerivativesandStateVariables.html
# https://python-forge.readthedocs.io/en/latest/signature.html
# https://realpython.com/python-kwargs-and-args/
# https://blogs.mathworks.com/simulink/2011/11/20/sample-time-offset/
# http://www.ece.northwestern.edu/local-apps/matlabhelp/toolbox/simulink/sfg/sfun_intro10a.html
# https://python-3-patterns-idioms-test.readthedocs.io/en/latest/StateMachine.html
# https://medium.com/swlh/implementing-finite-state-machine-as-python-language-native-construct-cb56e3c46215
# https://www.educba.com/python-event-handler/
# https://www.mathworks.com/help/simulink/block-libraries.html
# https://stackoverflow.com/questions/25295327/how-to-check-if-a-python-class-has-particular-method-or-not
# https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.dag.topological_sort.html
# https://realpython.com/python-eval-function/
# https://www.mathworks.com/help/matlab/ref/ode15s.html
# https://www.mathworks.com/help/matlab/ref/ode23t.html
# https://www.mathworks.com/help/matlab/math/solve-differential-algebraic-equations-daes.html
# https://www.mathworks.com/help/mpc/ug/swing-up-control-of-a-pendulum-using-nonlinear-model-predictive-control.html
# https://ctms.engin.umich.edu/CTMS/index.php?example=InvertedPendulum&section=SimulinkModeling
# https://ctms.engin.umich.edu/CTMS/index.php?aux=Activities_Pendulum
# http://www.ece.northwestern.edu/local-apps/matlabhelp/techdoc/ref/ode45.html
# https://www.mathworks.com/help/symbolic/solve-daes-using-mass-matrix-solvers.html
# https://stackoverflow.com/questions/24436663/emulating-matlabs-ode15s-in-python
# http://wiki.scipy.org/NumPy_for_Matlab_Users
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.html
# https://www.reddit.com/r/Python/comments/277uni/matlab_ode15s_equivalent/
# https://courses.engr.illinois.edu/ece530/fa2020/slides1112.pdf
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
# https://www.reddit.com/r/Python/comments/277uni/matlab_ode15s_equivalent/
# https://groups.google.com/g/comp.soft-sys.matlab/c/unbcOkqDYKs?pli=1