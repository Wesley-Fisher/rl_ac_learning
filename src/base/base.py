
class Properties:
    def __init__(self):
        raise NotImplementedError

class History:
    def __init__(self):

        self.state_0 = None
        self.sensed_state_0 = None
        self.value_0 = None
        self.action_0 = None
        self.action_prob_0 = None

        self.state_1 = None
        self.sensed_state_1 = None
        self.value_1 = None
        self.reward_1 = None

        self.return_val = None

class State:
    def __init__(self):
        raise NotImplementedError


class World:
    def __init__(self, properties, state):
        self.properties = properties
        self.state = state
    
    def get_physical_state(self):
        raise NotImplementedError
    
    def state_to_sensed(self, state):
        raise NotImplementedError
    
    def get_sensed_state(self):
        return self.state_to_sensed(self.get_physical_state())
    
    def reset_state(self):
        raise NotImplementedError
    
    def is_state_terminal(self):
        raise NotImplementedError
    
    def step(self, action):
        raise NotImplementedError
    
    def get_reward(self, s0, s1, a, ap):
        raise NotImplementedError
    
    def animate(self, history):
        raise NotImplementedError