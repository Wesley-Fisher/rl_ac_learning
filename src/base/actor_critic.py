import numpy as np

import tensorflow.compat.v1.keras.backend as K
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, ReLU, Input, Softmax
from tensorflow.keras import initializers
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.regularizers import l1, l2

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

class ActorCritic:

    def __init__(self, settings):
        ik = initializers.RandomNormal(stddev=0.1, seed=1)
        ib = initializers.RandomNormal(stddev=0.1, seed=2)

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
        for i, num in enumerate(settings.shared_layers):
            layer = Dense(num, kernel_initializer=ik, bias_initializer=ib, name='dense_'+str(i))(layer)
            layer = ReLU(name='dense_relu'+str(i))(layer)

        # CRITIC
        critic_layer = layer
        for i, num in enumerate(settings.critic_layers):
            critic_layer = Dense(num, kernel_initializer=ik, bias_initializer=ib, name='critic_dense_'+str(i))(critic_layer)
            critic_layer = ReLU(name='critic_dense_relu'+str(i))(critic_layer)
        critic_out = Dense(1, name='critic_out')(critic_layer)


        # ACTOR
        actor_layer = layer
        for i, num in enumerate(settings.actor_layers):
            actor_layer = Dense(num, kernel_initializer=ik, bias_initializer=ib, name='actor_dense_'+str(i))(actor_layer)
            actor_layer = ReLU(name='actor_dense_relu'+str(i))(actor_layer)
        actor_out = Dense(settings.actor_shape, name='actor_out')(actor_layer)
        actor_out = Softmax(name='actor_softmax')(actor_out)


        outputs = [critic_out, actor_out]

        # MODEL

        def actor_critic_loss(loss_critic_out, loss_actor_out, loss_return_value, loss_action, loss_advantage):
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
            actor_loss = -tf.math.reduce_sum(adjustments)

            # CRITIC LOSS: match return value
            critic_loss = K.pow(loss_return_value - loss_critic_out, 2)

            # ENTROY LOSS
            entropy = tf.math.reduce_sum(loss_actor_out * K.log(loss_actor_out))

            return settings.k_actor*actor_loss + settings.k_critic*critic_loss + settings.k_entropy*entropy

        self.model = Model(inputs=inputs, outputs=outputs, name='actor_critic')
        optimizer = Adam(settings.alpha)
        self.model.add_loss(actor_critic_loss(critic_out, actor_out, target_input, action_input, advantage_input))
        self.model.compile(optimizer=optimizer)
