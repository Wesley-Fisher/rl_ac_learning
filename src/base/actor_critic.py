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
        self.shared_layers = None
        self.actor_layers = None
        self.critic_layers = None
        self.alpha = None

        self.k_actor = 1.0
        self.k_critic = 1.0
        self.k_entropy = 0.0
        self.dropout = 0.0

def actor_critic_loss(loss_critic_out, loss_actor_out, loss_return_value, loss_action, loss_advantage, k_critic, k_actor, k_entropy):
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
            eps = 1e-4
            loss_actor_out = K.clip(loss_actor_out, eps, 1.0-eps) # Prevent 0 or 1

            log_prob = K.log(loss_actor_out)
            selected_act = loss_action * log_prob
            adjustments = selected_act * loss_advantage
            actor_loss = -K.sum(adjustments)

            critic_loss = K.pow(loss_return_value - loss_critic_out, 2)

            entropy = tf.math.reduce_sum(loss_actor_out * K.log(loss_actor_out))

            return k_actor*actor_loss + k_critic*critic_loss + k_entropy*entropy

class ActorCritic:

    def __init__(self, settings):
        self.settings = settings
        ik = initializers.RandomNormal(stddev=0.5, seed=1)
        ib = initializers.RandomNormal(stddev=0.01, seed=2)

        # INPUTS
        state_input = Input(shape=settings.in_shape, name='state_in')
        action_input = Input(shape=settings.actor_shape, name='action_in')
        target_input = Input(shape=(1), name='target_in')
        advantage_input = Input(shape=(1), name='advantage_in')

        inputs = [state_input, action_input, target_input, advantage_input]

        # NULL INPUTS (for prediction)
        self.null_action = np.array([[0.0]*settings.actor_shape])
        self.null_target = np.array([0.0])
        self.null_advantage = np.array([0.0])

        # SHARED LAYERS
        layer = state_input
        shared_layers = [layer]
        for i, num in enumerate(settings.shared_layers):
            layer = Dense(num, activation='sigmoid', kernel_initializer=ik, bias_initializer=ib, name='dense_'+str(i))(shared_layers[-1])
            shared_layers.append(layer)
            layer = LeakyReLU(alpha=0.3, name='dense_relu'+str(i))(shared_layers[-1])
            shared_layers.append(layer)
            #layer = Dropout(settings.dropout, name='dense_dropout'+str(i))(shared_layers[-1])
            #shared_layers.append(layer)
        
        # CRITIC
        critic_layer = shared_layers[-1]
        critic_layers = [critic_layer]
        for i, num in enumerate(settings.critic_layers):
            critic_layer = Dense(num, kernel_initializer=ik, bias_initializer=ib, name='critic_dense_'+str(i))(critic_layers[-1])
            critic_layers.append(critic_layer)
            critic_layer = LeakyReLU(alpha=0.3, name='critic_dense_relu'+str(i))(critic_layers[-1])
            critic_layers.append(critic_layer)
            #critic_layer = Dropout(settings.dropout, name='critic_dense_dropout'+str(i))(critic_layers[-1])
            #critic_layers.append(critic_layer)
        critic_out = Dense(1, name='critic_out')(critic_layer)


        # ACTOR
        actor_layer = shared_layers[-1]
        actor_layers = [actor_layer]
        for i, num in enumerate(settings.actor_layers):
            actor_layer = Dense(num, kernel_initializer=ik, bias_initializer=ib, name='actor_dense_'+str(i))(actor_layers[-1])
            actor_layers.append(actor_layer)
            actor_layer = ReLU(name='actor_dense_relu'+str(i))(actor_layers[-1])
            actor_layers.append(actor_layer)
            #actor_layer = Dropout(settings.dropout, name='actor_dense_dropout'+str(i))(actor_layers[-1])
            #actor_layers.append(actor_layer)
        actor_out = Dense(settings.actor_shape, activation='softmax', name='actor_out')(actor_layers[-1])
        #actor_out = Softmax(name='actor_softmax')(actor_out)


        outputs = [critic_out, actor_out]

        # MODEL

        def actor_critic_loss(loss_actor_out, loss_action, loss_advantage):
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
            eps = 1e-4
            loss_actor_out = K.clip(loss_actor_out, eps, 1.0-eps) # Prevent 0 or 1

            log_prob = K.log(loss_actor_out)
            selected_act = loss_action * log_prob
            adjustments = selected_act * loss_advantage
            actor_loss = -K.sum(adjustments)


            return settings.k_actor*actor_loss

        def critic_loss(loss_critic_out, loss_return_value):
            critic_loss = K.pow(loss_return_value - loss_critic_out, 2)
            return tf.math.reduce_sum(settings.k_critic*critic_loss)
        
        def entropy_loss(loss_actor_out):
            entropy = tf.math.reduce_sum(loss_actor_out * K.log(loss_actor_out))
            return tf.math.reduce_sum(settings.k_entropy*entropy)

        self.model = Model(inputs=inputs, outputs=outputs, name='actor_critic')
        self.optimizer = Adam(settings.alpha)
        #self.model.add_loss(actor_critic_loss(actor_out, action_input, advantage_input))
        self.model.add_loss(critic_loss(critic_out, target_input))
        self.model.add_loss(entropy_loss(actor_out))
        self.model.compile(optimizer=self.optimizer, run_eagerly=True)
        '''
        self.model.compile(optimizer=optimizer,
                           loss = {'critic_out': critic_loss(critic_out, target_input),
                                   'actor_out': actor_critic_loss(actor_out, action_input, advantage_input)
                           })
        '''
        self.model.summary()

    def fit(self, data):
        for i in range(0, data[0].shape[0]):
            state = data[0][i]
            action = data[1][i]
            target = data[2][i]
            advantage = data[3][i]

            with tf.GradientTape() as tape:
                pred = self.model((state.reshape((1,2)), action, target, advantage))
                loss = actor_critic_loss(pred[0][0], pred[1][0], target, action, advantage, self.settings.k_critic, self.settings.k_actor, self.settings.k_entropy)

            gradients = tape.gradient(loss, self.model.trainable_weights)
            '''
            for i in range(0, len(gradients)):
                print(self.model.layers[i].name)
                print(gradients[i])
            '''
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))