import random
import numpy as np
from tensorflow.python.keras.engine import training

from base.base import Properties, State, World
from base.actor_critic import ActorCritic, NetworkSettings
from base.trainer import Trainer, TrainingSettings

class FourArmBanditProperties(Properties):
    def __init__(self):
        self.mu_sel = 2
        self.mu_unsel = -2

class FourArmBanditState(State):
    def __init__(self):
        self.in_state = [0.0, 0.0]
        self.acted = False

class FourArmBanditWorld(World):
    def __init__(self, properties, state):
        super(FourArmBanditWorld, self).__init__(properties, state)

    def is_state_terminal(self):
        return self.state.acted
    
    def reset_state(self):
        eps = 1e-3
        self.state.in_state = [random.randint(0, 1)+eps, random.randint(0, 1)+eps]
        self.state.acted = False
    
    def get_physical_state(self):
        return self.state
    
    def state_to_sensed(self, state):
        s = state.in_state
        return np.array([float(s[0]), float(s[1])]).reshape((2,))
    
    def step(self, action):
        self.state.acted = True
    
    def get_reward(self, s0, s1, a, ap):
        i = a.index(max(a))
        iTrue = s0.in_state[0] + 2*s0.in_state[1]
        if i == iTrue:
            return random.gauss(self.properties.mu_sel, 0.5)
        else:
            return random.gauss(self.properties.mu_unsel, 0.5)
    
    def animate(self, history, network):
        actions = []
        for h in history:
            print("%s -> %s/%s -> %s vs pred(%s) -> adv(%s)" %
                                     (str(h.state_0.in_state),
                                      str(h.action_prob_0),
                                      str(h.action_0),
                                      str(h.reward_1),
                                      str(h.value_0),
                                      str(h.advantage)
                                      ))

        for two in [0.0, 1.0]:
            for unit in [0.0, 1.0]:
                i = unit + 2*two

                vec = np.array([np.array([[unit], [two]]).reshape(2,)])
                data = (vec)
                acts = network.predict_actions(data)


                print("%s - (%s) - (%s)" % (str(acts), str(acts.index(max(acts))), str(vec)))
        print("")

if __name__ == "__main__":
    N_batches = 500
    N_per_batch = 100

    properties = FourArmBanditProperties()
    state = FourArmBanditState()
    world = FourArmBanditWorld(properties, state)

    network_settings = NetworkSettings()
    network_settings.in_shape = (2,)
    network_settings.actor_shape = 4
    network_settings.shared_layers = [8, 4]
    network_settings.actor_layers = [8, 4]
    network_settings.critic_layers = [4, 4]
    network_settings.alpha = 1e-4
    network_settings.k_actor = 1e0
    network_settings.k_critic = 1e0
    network_settings.k_entropy = 1e-6
    network_settings.dropout = 0

    actor_critic = ActorCritic(network_settings)

    training_settings = TrainingSettings()
    training_settings.gamma = 0.9
    training_settings.exploration = 0.05

    trainer = Trainer(world, actor_critic, training_settings)

    for i in range(0, N_batches):
        batch_hist = trainer.run_batch(N_per_batch)

        if len(batch_hist) > 0:
            world.animate(batch_hist[-1], actor_critic)
        
        trainer.train_on_batch(batch_hist, 0, 1)

