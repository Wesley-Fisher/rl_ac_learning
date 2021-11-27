from copy import Error
import random
import numpy as np

from .base import History

class TrainingSettings:
    def __init__(self):
        self.gamma = None
        self.exploration = None

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
            data = (np.array([h.sensed_state_0]),
                    self.actor_critic.null_action,
                    self.actor_critic.null_target,
                    self.actor_critic.null_advantage)
            '''
            print(data)
            print(data[0].shape)
            '''
            pred = self.actor_critic.model.predict(data)

            h.value_0 = float(pred[0][0])

            acts = pred[1][0].tolist()
            ai = acts.index(max(acts))

            # Add Exploration
            if random.uniform(0.0, 1.0) < self.settings.exploration:
                L = len(acts)
                exp = random.randint(0, L-1)
                i0 = ai
                ai = (ai + exp) % L
                i1 = ai
                #print("(%f + %f) mod %f = %f" % (i0, exp, L, i1))

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
            data = (np.array([h.sensed_state_1]),
                    self.actor_critic.null_action,
                    self.actor_critic.null_target,
                    self.actor_critic.null_advantage)
            pred = self.actor_critic.model.predict(data)
            h.value_1 = float(pred[0][0])

            h.reward_1 = self.world.get_reward(h.state_0,
                                               h.state_1,
                                               h.action_0,
                                               h.action_prob_0)
            '''
            print(h.reward_1)
            '''
            hist.append(h)
        
        if len(hist) == 0:
            return []
        
        # Process returns
        hist[-1].return_val = hist[-1].reward_1
        for i in range(len(hist)-2, -1, -1):
            hist[i].return_val = hist[i].reward_1 + self.settings.gamma * hist[i+1].return_val
        
        for i in range(0, len(hist)):
            hist[i].advantage = hist[i].reward_1 + self.settings.gamma*hist[i].value_1 - hist[i].value_0

        return hist
    
    def train_on_batch(self, batch_hist, verbosity, num_epochs=1):
        if len(batch_hist) == 0:
            return
        
        samples = []
        for batch in batch_hist:
            samples = samples + batch
        
        states = np.array([s.sensed_state_0 for s in samples])
        advantages = np.array([s.advantage for s in samples])
        targets = advantages
        actions = np.array([s.action_0 for s in samples])

        data = (states, actions, targets, advantages)
        #print(data)

        '''
        print(states.shape)
        print(actions.shape)
        print(targets.shape)
        print(advantages.shape)
        '''
        '''
        for state, adv, target, act in zip (states, advantages, targets, actions):
            print("%s -> %s -> %s / %s" % (str(state), str(act), str(target), str(adv)))
        '''
        self.actor_critic.fit(data)
        '''
        self.actor_critic.model.fit(x=data,
                                    verbose=verbosity,
                                    batch_size=1,
                                    epochs=num_epochs)
        '''

        #for layer in self.actor_critic.model.layers: print(layer.get_config()['name'], '\n', layer.get_weights())
