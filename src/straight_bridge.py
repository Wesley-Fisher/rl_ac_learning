import random
import copy
import numpy as np
from tensorflow.python.keras.engine import training

from base.base import Properties, State, World
from base.actor_critic import ActorCritic, NetworkSettings
from base.trainer import Trainer, TrainingSettings

STEPS = 15

class StraightBridgeProperties(Properties):
    def __init__(self):
        pass

class StraightBridgeState(State):
    def __init__(self):
        self.position = [0.0] * STEPS
        self.off = False

class StraightBridgeWorld(World):
    def __init__(self, properties, state):
        super(StraightBridgeWorld, self).__init__(properties, state)

    def is_state_terminal(self):
        return self.state.off or self.state.position[STEPS-1] > 0.5
    
    def reset_state(self):
        eps = 0
        self.state.position = [0.0] * STEPS
        self.state.position[0] = 1.0
        self.state.off = False
    
    def get_physical_state(self):
        return copy.deepcopy(self.state)
    
    def state_to_sensed(self, state):
        return np.array(state.position)

    def get_pos(self, positions):
        return positions.index(max(positions))
    
    def step(self, action):
        i = action.index(max(action))
        if i == 0:
            pos = self.get_pos(self.state.position)
            if pos < STEPS:
                self.state.position[pos] = 0.0
                self.state.position[pos+1] = 1.0
            else:
                self.state.off = True
        elif i == 1:
            self.state.off = True
        elif i == 2:
            pos = self.get_pos(self.state.position)
            if pos > 0:
                self.state.position[pos] = 0.0
                self.state.position[pos-1] = 1.0
            else:
                self.state.off = True
        elif i == 3:
            self.state.off = True
    
    def get_reward(self, s0, s1, a, ap):
        if self.get_pos(s1.position) == STEPS - 1:
            return 1
        if s1.off:
            return -1
        return -0.01
    
    def animate(self, history, network):
        actions = []
        i0 = 0
        N = len(history)
        for h in history:
            i0 = i0 + 1
            print("(%d/%d): %s(%s) -> %s/%s -> %s vs pred(%s) -> adv(%s)" %
                                     (i0, N,
                                      str(h.state_0.position),
                                      str(h.state_0.off),
                                      str(h.action_prob_0),
                                      str(h.action_0),
                                      str(h.reward_1),
                                      str(h.value_0),
                                      str(h.advantage)
                                      ))

        vals = [0.0] * STEPS
        fors = [0.0] * STEPS
        for i in range(0, STEPS):
            position = [0.0]  * STEPS
            position[i] = 1.0
            vec = np.array([np.array(position)])
            data = (vec)

            val = network.predict_value(data)
            vals[i] = float('%.3f' % val)

            acts = network.predict_actions(data)
            fors[i] = float('%.3f' % acts[0])

        print(vals)
        print(fors)
        print("")

if __name__ == "__main__":
    N_batches = 500
    N_per_batch = 100

    properties = StraightBridgeProperties()
    state = StraightBridgeState()
    world = StraightBridgeWorld(properties, state)

    network_settings = NetworkSettings()
    network_settings.in_shape = (STEPS,)
    network_settings.actor_shape = 4
    network_settings.actor_layers = [8, 4]
    network_settings.critic_layers = [8, 4]
    network_settings.alpha = 1e-3
    network_settings.k_actor = 1e0
    network_settings.k_entropy = 1e-9
    network_settings.dropout = 0.25

    actor_critic = ActorCritic(network_settings)

    training_settings = TrainingSettings()
    training_settings.gamma = 0.9
    training_settings.exploration = 0.1

    trainer = Trainer(world, actor_critic, training_settings)

    for i in range(0, N_batches):
        batch_hist = trainer.run_batch(N_per_batch)

        if len(batch_hist) > 0:
            world.animate(batch_hist[-1], actor_critic)
        
        trainer.train_on_batch(batch_hist, 0, 1)

