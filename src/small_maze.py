import random
import copy
import numpy as np
from tensorflow.python.keras.engine import training

from base.base import Properties, State, World
from base.actor_critic import ActorCritic, NetworkSettings
from base.trainer import Trainer, TrainingSettings

WIDTH = 7
HEIGHT = 4

class SmallMazeProperties(Properties):
    def __init__(self):
        self.open = 0
        self.wall = 1
        # Index y first, then x
        self.maze = [[self.open, self.wall, self.open, self.open, self.open, self.open, self.open],
                     [self.open, self.wall, self.open, self.wall, self.open, self.wall, self.wall],
                     [self.open, self.wall, self.open, self.wall, self.open, self.wall, self.wall],
                     [self.open, self.open, self.open, self.wall, self.open, self.open, self.open],
                    ]
        self.start_y = 0
        self.start_x = 0
        self.end_y = 3
        self.end_x = 6

class SmallMazeState(State):
    def __init__(self):
        self.x = 0
        self.y = 0
        self.off = False
        self.time = 0
        self.hit = False

class SmallMazeWorld(World):
    def __init__(self, properties, state):
        super(SmallMazeWorld, self).__init__(properties, state)

    def is_state_terminal(self):
        if self.state.off or self.state.time > 25:
            return True
        if self.state.x == self.properties.end_x and self.state.y == self.properties.end_y:
            return True
        return False
    
    def reset_state(self):
        self.state.x = 0
        self.state.y = 0
        self.state.off = False
        self.state.time = 0
        self.state.hit = False
    
    def get_physical_state(self):
        return copy.deepcopy(self.state)
    
    def state_to_sensed(self, state):
        hor = [0.0] * WIDTH
        hor[state.x] = 1
        ver = [0.0] * HEIGHT
        ver[state.y] = 1

        right = [0.0]
        if state.x == WIDTH - 1 or self.properties.maze[state.y][state.x+1] == self.properties.wall:
            right = [1.0]
        up = [0.0]
        if state.y == 0 or self.properties.maze[state.y-1][state.x] == self.properties.wall:
            up = [1.0]
        left = [0.0]
        if state.x == 0 or self.properties.maze[state.y][state.x-1] == self.properties.wall:
            left = [1.0]
        down = [0.0]
        if state.y == HEIGHT - 1 or self.properties.maze[state.y+1][state.x] == self.properties.wall:
            down = [1.0]
        return np.array(hor + ver + right + up + left + down)
    
    def step(self, action):
        # Action: right, up, left, down
        self.state.hit = False
        i = action.index(max(action))
        
        x1 = self.state.x
        y1 = self.state.y
        off = False

        # Right
        if i == 0:
            if self.state.x < WIDTH-1:
                x1 = self.state.x + 1
                y1 = self.state.y
            else:
                off = True
        
        # Up
        if i == 1:
            if self.state.y > 0:
                x1 = self.state.x
                y1 = self.state.y - 1
            else:
                off = True
        
        # Left
        if i == 2:
            if self.state.x > 0:
                x1 = self.state.x - 1
                y1 = self.state.y
            else:
                off = True
        
        # Down
        if i == 3:
            if self.state.y < HEIGHT-1:
                x1 = self.state.x
                y1 = self.state.y + 1
            else:
                off = True
        
        if off:
            self.state.off = True
        elif self.properties.maze[y1][x1] == self.properties.wall:
            self.state.hit = True
        else:
            self.state.x = x1
            self.state.y = y1
            state.time = state.time + 1

        #print("(%d, %d, %d)" % (self.state.x, self.state.y, self.state.off))
    
    def get_reward(self, s0, s1, a, ap):
        if s1.off:
            return -5
        if s1.hit:
            return -1
        if s1.x == self.properties.end_x and s1.y == self.properties.end_y:
            return 10
        return -.1*((self.properties.end_x - s1.x) + (self.properties.end_y - s1.y))
    
    def animate(self, history, network):

        empty = []
        for y in range(0, HEIGHT):
            row = [0.0] * WIDTH
            empty.append(row)

        vals = copy.deepcopy(empty)
        actions = copy.deepcopy(empty)
        for h in history:
            p0 = "(%d, %d)" % (h.state_0.x, h.state_0.y)
            a = str(h.action_prob_0)
            act = "(%d)" % h.action_0.index(max(h.action_0))
            p1 = "(%d, %d)" % (h.state_1.x, h.state_1.y)
            end = "[%d, %d]" % (h.state_1.off, h.state_1.hit)
            r = "%.1f" % h.reward_1
            ret = "%.2f" % h.return_val
            adv = "%.2f" % h.advantage

            print("%s->%s%s->%s%s  =>  %s->%s / %s" % (p0, a, act, p1, end, r, ret, adv))

        for x in range(0, WIDTH):
            for y in range(0, HEIGHT):
                state = SmallMazeState()
                state.x = x
                state.y = y

                sensed = self.state_to_sensed(state)
                data = (np.array([np.array(sensed)]))

                val = network.predict_value(data)
                vals[y][x] = float('%.2f' % val)

                acts = network.predict_actions(data)
                actions[y][x] = acts.index(max(acts))

        for val in vals:
            print(val)
        print("")
        for action in actions:
            print(action)
        print("")
        print("")
        print("")

if __name__ == "__main__":
    N_batches = 500
    N_per_batch = 10

    properties = SmallMazeProperties()
    state = SmallMazeState()
    world = SmallMazeWorld(properties, state)

    network_settings = NetworkSettings()
    network_settings.in_shape = (WIDTH+HEIGHT+4,)
    network_settings.actor_shape = 4
    network_settings.actor_layers = [16, 16, 8]
    network_settings.critic_layers = [16, 8]
    network_settings.alpha = 1e-4
    network_settings.k_actor = 1e0
    network_settings.k_entropy = 1e-9
    network_settings.dropout = 0.25

    actor_critic = ActorCritic(network_settings)

    training_settings = TrainingSettings()
    training_settings.gamma = 0.8
    training_settings.exploration = 0.0
    training_settings.exploration_drawn = 0.1

    trainer = Trainer(world, actor_critic, training_settings)

    for i in range(0, N_batches):
        batch_hist = trainer.run_batch(N_per_batch)

        if len(batch_hist) > 0:
            world.animate(batch_hist[-1], actor_critic)
        
        trainer.train_on_batch(batch_hist, 0, 1)

