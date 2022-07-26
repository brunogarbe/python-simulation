from abc import ABC, abstractmethod
from enum import Enum
from inspect import signature
from scipy.optimize import fsolve
import numpy as np
import networkx as nx
from scipy.integrate import solve_ivp


def generate_track_event(call, idx, sz, sys, input_idx):
    """Generate an event function in the format that the solve_ivp function can track, from the
    way they are normally coded in the simulation library, from an method inside an class, and
    considering the transformation indexes that convert the ``X`` and ``Y`` global vectors from
    the system to the ``x`` and ``u`` local vectors of the object.
    
    Args:
        call (method) : event method from class
        idx (int) : index of the start of states from global to local
        sz (int) : number of states of the class
        sys (System) : System being simulated
        input_idx (list[int]) : list of indexes that transform global ``Y`` to local ``u``
        
    Returns:
        callable function with signature event(t, x)
    
    References:
        Simultaneous event detection using solve_ivp
        https://stackoverflow.com/questions/60136367/simultaneous-event-detection-using-solve-ivp
        
        scipy.integrate.solve_ivp
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
    """
    
    # create the event that will be returned
    def event_function(t, X):
        # calculates the global output vector ``Y`` for the system
        Y = sys.output(t, X)
        
        # gets the local input vector ``u`` and state vector ``x`` for the object
        #u = Y[input_idx] if len(input_idx) != 0 else None
        u = Y[input_idx]
        x = X[idx:idx+sz]
        
        return call(t, x, u)
    
    # Assign the attributes for the solve_ivp terminate if the event occurs and
    # the direction of the zero crossing, as the direction is positive, the event will only trigger
    # when going from negative to positive.
    event_function.terminal = True
    event_function.direction = 1
    
    return event_function
        
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
    
    def add(self, bl):
        self.blocks.append(bl)

    def connect(self, bl_out, out_port, bl_in, in_port):
        self.connections.append( (bl_out, out_port, bl_in, in_port) )

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
        self.block_order = []

        self.connections = []
        self.direct_connections = []

        self.outnames = []
        
        self.exit_flag = False

    def add(self, bl):

        if isinstance(bl, Subsystem) == True:
            #print('BINGO!!!')
            for b in bl.blocks:
                #print('bingo')
                self.blocks.append(b)

            # connect the internal
            for bl_out, out_port, bl_in, in_port in bl.connections:
                #print('con')
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
            self.add(bl_out)

        if bl_in not in self.blocks:
            self.add(bl_in)

        # check that the connections are not already used
        if bl_in.inputs[in_port].block is not None and bl_in.inputs[in_port].port is not None:
            raise Exception("Channel already used!!")
            
        bl_out.add_output_connection(out_port, bl_in, in_port)
        bl_in.set_input_connection(in_port, bl_out, out_port)

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

        #print(block_order)

        # calculate the state/output indexes
        self.num_states = 0
        self.num_outputs = 0
        idx_state = 0
        idx_output = 0
        self.block_order = []
        state_dict = {}
        output_dict = {}

        for bl_id in block_order:
            sys = block_id_dict[bl_id]
            self.num_states += sys.num_states
            self.num_outputs += sys.num_outputs
            for i in range(sys.num_outputs):
                self.outnames.append(sys.name + '.' + sys.outputs[i].name)
            self.block_order.append( (sys, idx_state, sys.num_states, idx_output, sys.num_outputs) )
            state_dict[sys] = (idx_state, sys.num_states)
            output_dict[sys] = (idx_output, sys.num_outputs)

            sys.state_index = (idx_state, sys.num_states)
            sys.output_index = (idx_output, sys.num_outputs)

            idx_state += sys.num_states
            idx_output += sys.num_outputs


        # calculate the input indexes
        for bl_id in block_order:
            sys = block_id_dict[bl_id]
            input_idx = []
            for in_port in range(sys.num_inputs):
                idx_in_out_matrix = output_dict[sys.inputs[in_port].block][0] + sys.inputs[in_port].port
                input_idx.append(idx_in_out_matrix)
            sys.input_index = input_idx

        # model, ev, handler, idx, sz in
        for i, _ in enumerate(self.all_events):
            model = self.all_events[i][0]
            idx_state, size_state = state_dict[model]
            self.all_events[i][3] = idx_state
            self.all_events[i][4] = size_state

        self.out = np.zeros(self.num_outputs)

    def initial(self):
        x0 = np.zeros(self.num_states)
        for sys, idx_states, num_states, _, _ in self.block_order:
            if sys.num_states > 0:
                x0[idx_states:idx_states+num_states] = sys.initial()

        return x0
        
    def derivative(self, t, x):
        dx = np.zeros(self.num_states)
        out = self.output(t, x)
        
        for bl, idx_state, num_states, _, _ in self.block_order:
            u = out[bl.input_index] if bl.num_inputs > 0 else None

            if num_states != 0:
                dx[idx_state:idx_state+num_states] = bl.derivative(t, x[idx_state:idx_state+num_states], u)

        return dx

    def output(self, t, x):
        out = np.zeros(self.num_outputs)
        
        for bl, idx_state, num_states, idx_out, num_outputs in self.block_order:
            if bl.is_direct_feedthrough == False:
                u = None
            else:
                u = out[bl.input_index] if (bl.num_inputs > 0) and (bl.is_direct_feedthrough) else None
            
            out[idx_out:idx_out+num_outputs] = bl.output(t, x[idx_state:idx_state+num_states], u)
        return out
        
    def exit(self, t, x, u):
        self.exit_flag = True

    def get_events_list(self):
        ret = []
        for model, ev, _, idx, sz in self.all_events:
            ret.append(generate_track_event(ev, idx, sz, self, model.input_index))
        return ret
        
    def get_event_indexes(self, idx):
        idx_state = self.all_events[idx][3]
        size_state = self.all_events[idx][4]
        return idx_state, size_state
        
    def get_event_handler(self, idx):
        handler = self.all_events[idx][2]
        return handler
        
    def get_event_block(self, idx):
        handler = self.all_events[idx][0]
        return handler

    def __str__(self):
        ret = '#####################################################################################\n'
        
        ret += '# Blocks:\n'
        for bl in self.blocks:
            ret += f'{"":3} {bl.name} {bl.id}\n'
        
        ret += '# Connections:\n'
        for bl, idx, num_states, idx_out, num_outputs in self.block_order:
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
            event = self.all_events[i][1]
            handler = self.all_events[i][2]
            idx_state = self.all_events[i][3]
            size_state = self.all_events[i][4]
            
            event_name = event.__self__.name + '.' + event.__name__
            handler_name = handler.__self__.name + '.' + handler.__name__
            ret += f'{i:3} {model.name:20} {event_name:30} {handler_name:30} [{idx_state}:{idx_state+size_state}]\n'
        
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

    def call_handlers(self, t, X, Y):
        for handler in self.events_table[0][1]:
            if handler is not None:
                x = None if handler.__self__.num_states == 0 else X[idx_start:idx_end]
                u = None if handler.__self__.num_inputs == 0 else Y[handler.__self__.input_index]
                idx_start = handler.__self__.state_index[0]
                idx_end = handler.__self__.state_index[0] + handler.__self__.state_index[1]
                #print(Y, handler.__self__.input_index)
                handler(t, x, u)
                
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

def simulate(sys, t_start=0.0, t_final=10.0, dt_max = np.inf):

    sys.prepare()

    if sys.num_states == 0:
        return simulate_nostates(sys)
    else:
        return simulate_scipy(sys, t_start, t_final, dt_max)


def simulate_nostates(sys):

    t_start = 0.0
    t_final=10.0
    dt_max = 0.1
    X_null = np.array([])

    # start the scheduler
    scheduler = Scheduler()
    scheduler.add(t_final, None)

    for bl in sys.block_order:
        if hasattr(bl[0], 'add_schedule') == True:
            bl[0].add_schedule(scheduler, 0.0, t_final)

    t_array = np.empty(0)
    x_array = np.empty((sys.num_states, 0))
    y_array = np.empty((sys.num_outputs, 0))
    events_array = []

    # TODO: check for events that happens before t_start

    t_interval_end = t_start
    while scheduler.has_ended() == False:
        t_interval_start = t_interval_end
        t_interval_end = scheduler.next_time()

        for t_cur in np.arange(t_interval_start, t_interval_end, dt_max):
            Y_cur = sys.output(t_cur, X_null)

            # TODO: check for events!!!
            
            t_array = np.append(t_array, t_cur)
            y_array = np.append(y_array, Y_cur.reshape((sys.num_outputs,1)), axis=1)

        scheduler.call_handlers(t_interval_end, X_null, np.array([0, 0, 0, 0, 0, 0, 0]))
        scheduler.pop()

    res = Results(time=t_array, outputs=y_array, names=sys.outnames)

    return res

def simulate_scipy(sys, t_start, t_final, dt_max):

    # set the history arrays for record time, state and output vectors for the system
    t_array = np.empty(0)
    x_array = np.empty((sys.num_states, 0))
    y_array = np.empty((sys.num_outputs, 0))

    # sets the global initial state vector
    X_initial = sys.initial()
    
    # start the scheduler and set an final time event with no event handler
    scheduler = Scheduler()
    scheduler.add(t_final, None)

    # check for time events defined in the systems blocks
    for bl in sys.block_order:
        if hasattr(bl[0], 'add_schedule') == True:
            bl[0].add_schedule(scheduler, 0.0, t_final)

    # Generate a list of events to track in the solve_ivp format.
    all_events = sys.get_events_list()
    
    # TODO: check for events out of the simulation range [t_start, t_final)

    # check for time event on the start time of the simulation and calls the handle if necessary
    if np.isclose(t_start, scheduler.next_time()) == True:
        Y_initial = sys.output(t_start, X_initial)
        scheduler.call_handlers(t_start, X_initial, Y_initial)
        scheduler.pop()

    # iterate over the regions [t_interval_start, t_interval_end), using solve_ivp to integrate on
    # the defined interval. The t_interval_start and t_interval_end are obtained from the t_start
    # value and the time values on the scheduler object.
    t_interval_end = t_start
    while scheduler.has_ended() == False:
        t_interval_start = t_interval_end
        t_interval_end = scheduler.next_time()
        
        # iterave over state events that occur in the time region being solved py solve_ivp
        while True:
        
            # Solve an initial value problem for a system of ODEs.
            sol = solve_ivp(sys.derivative, (t_interval_start, t_interval_end), X_initial, events=all_events, max_step=dt_max)
            
            # The solver successfully reached the end of tspan.
            if sol.status == 0:
                
                # get the size of the solution from solve_ivp
                num_samples = sol.y.size // len(sol.y)
                
                # get the simulated state from solve_ivp solution and calculates the output vector
                y_sol = np.empty((sys.num_outputs, num_samples))
                for i in range(num_samples):
                    y_sol[:,i] = sys.output(sol.t[i], sol.y[:,i])
                
                # append the results to history arrays
                t_array = np.append(t_array, sol.t)
                x_array = np.append(x_array, sol.y, axis=1)
                y_array = np.append(y_array, y_sol, axis=1)
                
                # stops the simulation in the time region, since the solver reaches the end of the interval
                break

            # A termination event occurred.
            elif sol.status == 1:

                # set the new start interval for the simulation
                t_interval_start = sol.t_events[0]
                X_initial = sol.y_events[0][0,:]
                Y_initial = sys.output(t_interval_start, X_initial)
                
                # Iterate over all state events and check if they are triggered since
                # solve_ivp does not detect additional events that happen at the same time
                # https://stackoverflow.com/questions/60136367/simultaneous-event-detection-using-solve-ivp
                for i, event_function in enumerate(all_events):
                    idx, sz = sys.get_event_indexes(i)
                    handler = sys.get_event_handler(i)
                    bl = sys.get_event_block(i)
                    value = event_function(t_interval_start, X_initial)
                    
                    if np.isclose(value, 0.0) == True:
                        u = Y_initial[bl.input_index]
                        ret = handler(t_interval_start, X_initial[idx:sz], u)
                        if ret is not None:
                            X_initial[idx:sz] = ret 
                
                # get the size of the solution from solve_ivp
                num_samples = sol.y.size // len(sol.y)

                # get the simulated state from solve_ivp solution and calculates the output vector
                y_sol = np.empty((sys.num_outputs, num_samples))
                for i in range(num_samples):
                    y_sol[:,i] = sys.output(sol.t[i], sol.y[:,i])

                # append the results to history arrays
                t_array = np.append(t_array, sol.t)
                x_array = np.append(x_array, sol.y, axis=1)
                y_array = np.append(y_array, y_sol, axis=1)
                
                # terminate the simulation if is exit_flag is set True
                if sys.exit_flag is True:
                    break
            
            # solve_ivp integration step failed
            elif sol.status == -1:
                raise Exception(f"solve_ivp integration step failed: {sol.message}")
                
            else:
                raise Exception(f"Unknown solve_ivp status for algorithm termination: {sol.message}")
        
        # Sets the new initial X state for the next time region
        X_initial = sol.y[:,-1]
        
        # Call the time event handlers of the end of the time region being simulated
        scheduler.call_handlers(t_interval_end, X_initial, y_array[:,-1])
        scheduler.pop()

        # Breaks the second iteration loop
        if sys.exit_flag is True:
            break

    # Prepare the results output object and return it
    res = Results(time=t_array, outputs=y_array, names=sys.outnames)
    return res