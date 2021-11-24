from copy import Error
import random
import numpy as np

from .base import History

class TrainingSettings:
    def __init__(self):
        self.gamma = None

class Trainer:
    def __init__(self, world, actor_critic, settings):
        self.world = world
        self.actor_critic = actor_critic
        self.settings = settings
    
    def run_batch(self, N):
        batch_hist = []
        for i in range(0, N):
            hist = self.run_episode()
            if len(hist) > 0:
                batch_hist.append(hist)
        return batch_hist
    
    def run_episode(self):
        self.world.reset_state()
        hist = []

        length = 0
        while not self.world.is_state_terminal():
            length = length + 1
            h = History()

            # Sample at start
            h.state_0 = self.world.get_physical_state()
            h.sensed_state_0 = self.world.get_sensed_state()

            # Get Network Output
            data = (h.sensed_state_0,
                    self.actor_critic.null_action,
                    self.actor_critic.null_target,
                    self.actor_critic.null_advantage)
            pred = self.actor_critic.model.predict(data)

            h.value_0 = pred[0][0]

            acts = pred[1][0].tolist()
            ai = acts.index(max(acts))
            actions = [0.0 for ap in acts]
            actions[ai] = 1.0
            h.action_0 = actions
            h.action_prob_0 = acts

            # Apply Physics
            self.world.step(h.action_0)

            # Sample at end
            h.state_1 = self.world.get_physical_state()
            h.sensed_state_1 = self.world.get_sensed_state()

            # Get Network Output
            data = (h.sensed_state_1,
                    self.actor_critic.null_action,
                    self.actor_critic.null_target,
                    self.actor_critic.null_advantage)
            pred = self.actor_critic.model.predict(data)
            h.value_1 = pred[0][0]

            h.reward_1 = self.world.get_reward(h.state_0,
                                               h.state_1,
                                               h.action_0,
                                               h.action_prob_0)
            hist.append(h)
        
        if len(hist) == 0:
            return []
        
        # Process returns
        hist[-1].return_val = hist[-1].reward_1
        for i in range(len(hist)-2, -1, -1):
            hist[i].return_val = hist[i].reward_1 + self.settings.gamma * hist[i+1].return_val
        
        return hist
    
    def train_on_batch(self, batch_hist, verbosity):
        if len(batch_hist) == 0:
            return
        
        samples = []
        for batch in batch_hist:
            samples = samples + batch
        
        states = np.array([s.sensed_state_0 for s in samples])
        advantages = np.array([s.reward_1 + self.settings.gamma*s.value_1 - s.value_0 for s in samples])
        targets = advantages
        actions = np.array([s.action_0 for s in samples])

        '''
        print(states.shape)
        print(actions.shape)
        print(targets.shape)
        print(advantages.shape)
        '''
        self.actor_critic.model.fit(x=(states, actions, targets, advantages),
                                    verbose=verbosity,
                                    batch_size=1)
