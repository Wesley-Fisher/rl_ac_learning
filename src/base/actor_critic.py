import numpy as np

import tensorflow.compat.v1.keras.backend as K
import tensorflow as tf
#tf.compat.v1.disable_eager_execution()
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, ReLU, Input, Softmax, Dropout, LeakyReLU
from tensorflow.keras import initializers
#from tensorflow.keras.optimizers import Adam, SGD
from keras.optimizers import Adam
from tensorflow.keras.regularizers import l1, l2
from keras.utils.vis_utils import plot_model

class NetworkSettings:
    def __init__(self):
        self.in_shape = None
        self.actor_shape = None
        self.actor_layers = None
        self.critic_layers = None
        self.alpha = None

        self.k_actor = 1.0
        self.k_entropy = 0.0
        self.dropout = 0.0

def actor_loss(loss_actor_out, loss_action, loss_advantage, k_actor, k_entropy):
            # Heavy Influence: https://www.tensorflow.org/tutorials/reinforcement_learning/actor_critic
            '''
            ACTOR LOSS: change probability of action taken in direction of sign(advantage)
            critic_out: prediction of network
            actor_out: prediction of network. List of probabilities that sum to 1
            action: actions actually taken (list of 0's and 1's)

            Use log probabilities in loss function. We will decrease the loss
            We want the taken action (if advantage is positive) to be more likely
            - prob increases (0 to 1)
            - log(prob) increases (-inf to 0)
            - action * log_prob selects only the log_prob of the action taken
            - action * log_prob increases
            - action * log_prob * (advantage > 0) increases
            - action * log_prob * (advantage > 0) * (-1) decreases
            - we get closer to where we want to go
            Advantage:
            - multiply by advantage to control direction we want to go
            - positive advantage: we do want to increase probability, etc
            '''
            eps = 1e-3
            loss_actor_out = K.clip(loss_actor_out, eps, 1.0-eps) # Prevent 0 or 1

            log_prob = K.log(loss_actor_out)
            selected_act = loss_action * log_prob
            adjustments = selected_act * loss_advantage
            actor_loss = -K.sum(adjustments)

            entropy = tf.math.reduce_sum(loss_actor_out * K.log(loss_actor_out))

            '''
            print("Actor Loss:")
            print(loss_actor_out)
            print(log_prob)
            print(loss_action)
            print(selected_act)
            print(adjustments)
            print(actor_loss)
            print(entropy)
            '''

            return k_actor*actor_loss + k_entropy*entropy

class ActorCritic:

    def __init__(self, settings):
        self.settings = settings
        ik = initializers.RandomNormal(stddev=0.5, seed=1)
        ib = initializers.RandomNormal(stddev=0.01, seed=2)


        # CRITIC
        critic_input = Input(shape=settings.in_shape, name='critic_state_in')
        critic_layers = [critic_input]
        for i, num in enumerate(settings.critic_layers):
            critic_layer = Dense(num, kernel_initializer=ik, bias_initializer=ib, name='critic_dense_'+str(i))(critic_layers[-1])
            critic_layers.append(critic_layer)
            critic_layer = LeakyReLU(alpha=0.3, name='critic_dense_relu'+str(i))(critic_layers[-1])
            critic_layers.append(critic_layer)
            critic_layer = Dropout(settings.dropout, name='critic_dense_dropout'+str(i))(critic_layers[-1])
            critic_layers.append(critic_layer)
        critic_out = Dense(1, name='critic_out')(critic_layer)

        self.critic_model = Model(critic_input, critic_out, name='critic')
        self.optimizer = Adam(settings.alpha)
        self.critic_model.compile(optimizer=self.optimizer, run_eagerly=True, loss='mse')
        self.critic_model.summary()


        # ACTOR
        actor_input = Input(shape=settings.in_shape, name='actor_state_in')
        actor_layers = [actor_input]
        for i, num in enumerate(settings.actor_layers):
            actor_layer = Dense(num, kernel_initializer=ik, bias_initializer=ib, name='actor_dense_'+str(i))(actor_layers[-1])
            actor_layers.append(actor_layer)
            actor_layer = LeakyReLU(alpha=0.3, name='actor_dense_relu'+str(i))(actor_layers[-1])
            actor_layers.append(actor_layer)
            actor_layer = Dropout(settings.dropout, name='actor_dense_dropout'+str(i))(actor_layers[-1])
            actor_layers.append(actor_layer)
        actor_out = Dense(settings.actor_shape, activation='softmax', name='actor_out')(actor_layers[-1])

        self.actor_model = Model(actor_input, actor_out, name='actor')
        self.actor_model.compile(optimizer=self.optimizer, run_eagerly=True)
        self.actor_model.summary()
    
    def predict_value(self, data):
        return float(self.critic_model(data))
    
    def predict_actions(self, data):
        return self.actor_model(data).numpy().tolist()[0]

    def fit(self, data):

        # Critic is easy to train
        self.critic_model.fit(data[0], data[2], verbose=0)

        for i in range(0, data[0].shape[0]):
            state = data[0][i]
            action = data[1][i]
            target = data[2][i]
            advantage = data[3][i]

            with tf.GradientTape() as tape:
                pred = self.actor_model((state.reshape((1,self.settings.in_shape[0]))))
                loss = actor_loss(pred, action, advantage, self.settings.k_actor, self.settings.k_entropy)

            gradients = tape.gradient(loss, self.actor_model.trainable_weights)

            '''
            for i in range(0, len(gradients)):
                print(self.actor_model.layers[i].name)
                print(gradients[i])
            '''

            #for layer in self.actor_critic.model.layers: print(layer.get_config()['name'], '\n', layer.get_weights())
            
            self.optimizer.apply_gradients(zip(gradients, self.actor_model.trainable_weights))
