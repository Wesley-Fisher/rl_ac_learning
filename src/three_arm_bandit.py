import random
import numpy as np

from base.base import Properties, State, World
from base.actor_critic import ActorCritic, NetworkSettings
from base.trainer import Trainer, TrainingSettings

class ThreeArmBanditProperties(Properties):
    def __init__(self):
        self.mu  = [0, -0.5,  1]
        self.sig = [3, 0.1, 0.5]

class ThreeArmBanditState(State):
    def __init__(self):
        self.acted = False

class ThreeArmBanditWorld(World):
    def __init__(self, properties, state):
        super(ThreeArmBanditWorld, self).__init__(properties, state)

    def is_state_terminal(self):
        return self.state.acted
    
    def reset_state(self):
        self.state.acted = False
    
    def get_physical_state(self):
        return None
    
    def state_to_sensed(self, state):
        return np.array([0.0])
    
    def step(self, action):
        self.state.acted = True
    
    def get_reward(self, s0, s1, a, ap):
        i = ap.index(max(ap))
        mu = self.properties.mu[i]
        sig = self.properties.sig[i]
        return random.gauss(mu, sig)
    
    def animate(self, history):
        for h in history:
            print("%s - %s - %s" % (str(h.action_prob_0), str(h.action_0), str(h.value_0)))


if __name__ == "__main__":
    N_batches = 500
    N_per_batch = 100

    properties = ThreeArmBanditProperties()
    state = ThreeArmBanditState()
    world = ThreeArmBanditWorld(properties, state)

    network_settings = NetworkSettings()
    network_settings.in_shape = (1,)
    network_settings.actor_shape = 3
    network_settings.shared_layers = [3]
    network_settings.actor_layers = [3]
    network_settings.critic_layers = [2]
    network_settings.alpha = 1e-3
    network_settings.k_actor = 1.0
    network_settings.k_critic = 1.0
    network_settings.k_entropy = 0.0

    actor_critic = ActorCritic(network_settings)

    training_settings = TrainingSettings()
    training_settings.gamma = 0.9

    trainer = Trainer(world, actor_critic, training_settings)

    for i in range(0, N_batches):
        batch_hist = trainer.run_batch(N_per_batch)

        if len(batch_hist) > 0:
            world.animate(batch_hist[-1])
        
        trainer.train_on_batch(batch_hist, 0)

