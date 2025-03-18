import copy

import gym
from gym import spaces
import matplotlib.animation as animation
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["animation.html"] = "jshtml"
plt.rcParams['figure.dpi'] = 150  


STATES = ['O', 'x', 'x', 'x', 'x',
          'x', 'x', 'x', 'x', 'x',
          'x', 'x', 'C', 'x', 'P1',
          'x', 'x', 'x', 'x', 'x',
          'x', 'x', 'x', 'x', 'x']

REWARD_MAP = {'O': {'p': (1.0,), 'r': (-1,)},
              'x': {'p': (1.0,), 'r': (-1,)},
              's': {'p': (1.0,), 'r': (-1,)},
              'C': {'p': (1.0,), 'r': (-10,)},
              'P1': {'p': (1.0,), 'r': (10,)},
              'P2': {'p': (1.0,), 'r': (10,)}}

IMAGE_MAP = {'A': 'imgs/ansatt.jpg',
             'O': 'imgs/kontor_small.jpg',
             'x': 'imgs/gulv_small.jpg',
             's': 'imgs/kaffe_small.jpg',
             'C': 'imgs/byggearbeid_small.jpg',
             'P1': 'imgs/printer_small.jpg',
             'P2': 'imgs/printer_small.jpg'}

DONE_MAP = {'O': False, 'x': False, 's': False, 'C': True, 'P1': True, 'P2': True}


class OfficeEnv(gym.Env):
    """O is starting point, P1, P2 printers,
    C construction zones, s slippery spots.

    When mode == 'easy', the map looks as follows
        O   x  x  x   x
        x   x   s   x   x
        x   x   C   x   P1
        x   x   s   x   x
        x   x   x   x   x

    When mode == 'hard', the map looks as follows
        O   x  P2  x   x
        x   x   s   x   x
        x   x   C   x   P1
        x   x   s   x   x
        x   x   x   x   x

    Action space:       0: up, 1: down, 2: left, 3: right
                        Choosing an action that moves the agent into
                        a wall results in the agent staying in the same
                        spot and receiving the corresponding reward.
                        If a spot is slippery, there is a 1/3 probability of the
                        agent moving in the intended direction, and a 1/3 probability
                        for moving in either of the two perpendicular directions, respectively.
    Observation space:  Integer value in [0, 24] maps to a position in the
                        office space according to the flattened indices: 0 -> (0, 0), 1 -> (0, 1), 2 -> (0, 2), ...
                        The start state is always 0.
    Rewards:            r(C) = -10, r(O) = -1, r(s) = -1, r(x) = -1
                        In easy mode, printers cannot malfunction:
                            r(P1) = 5
                        In hard mode, printers can malfunction:
                            r(P1) = 5 w prob 90%, r(P1) = -10 w prob 10%
                            r(P2) = 5 w prob 50%, r(P2) = -10 w prob 50%
    End criteria:       End if agent moves into a spot with C, P1 or P2, or if
                        the number of steps taken exceeds max_steps.
        
    """
    metadata = {'render.modes': ['human', 'ascii']}

    def __init__(self, mode='easy', max_steps=50):
        super(OfficeEnv, self).__init__()
        self.states = STATES
        self.reward_map = copy.deepcopy(REWARD_MAP)
        self.image_map = copy.deepcopy(IMAGE_MAP)
        self.done_map = copy.deepcopy(DONE_MAP)
        self.max_steps = max_steps
        self.nrows = 5
        self.ncols = 5

        if mode == 'hard':
            self.states[2] = 'P2'
            self.states[7] = 's'
            self.states[17] = 's'
            self.reward_map['P1'] = {'p': (0.9, 0.1), 'r': (10, -10)}
            self.reward_map['P2'] = {'p': (0.3, 0.7), 'r': (10, -10)}
            self.is_slippery = True
        else:
            self.is_slippery = False

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(self.nrows*self.ncols)
        self.current_state = 0
        self.done = False
        self.steps = 0
        self.seed()

    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def step(self, action):
        self.steps += 1
        state = self._next_state(self.current_state, action)
        reward = self._state_to_reward(state)
        self.current_state = state
        observation = state
        done = self._is_done(state)
        info = {}
        return observation, reward, done, info

    def reset(self, seed=None):
        self.seed(seed)
        self.current_state = 0
        self.done = False
        self.steps = 0
        return self.current_state

    def render(self, mode='human'):
        if mode == 'ascii':
            printstring = ''
            for row in range(self.nrows):
                for col in range(self.ncols):
                    state = self._row_col_to_state(row, col)
                    if state == self.current_state:
                        printstring += 'A\t'
                    else:
                        printstring += f'{self.states[state]}\t'
                printstring += '\n'
            print(printstring)
            return printstring
        elif mode == 'human':
            compimage = np.zeros((180*self.nrows, 180*self.ncols, 3), dtype=int)
            for row in range(self.nrows):
                rowslice = slice(180*row, 180*(row+1))
                for col in range(self.ncols):
                    colslice = slice(180*col, 180*(col+1))
                    state = self._row_col_to_state(row, col)
                    if state == self.current_state:
                        compimage[rowslice, colslice] = self._get_agent_image().astype(int)
                    else:
                        compimage[rowslice, colslice] = self._get_state_image(state).astype(int)
            
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(1, 1, 1)
            ax.imshow(compimage)
            ax.set_xticks([])
            ax.set_yticks([])
            return compimage
        return None

    def close(self):
        pass

    def showmap(self, mode='human'):
        if mode == 'ascii':
            printstring = ''
            for row in range(self.nrows):
                for col in range(self.ncols):
                    state = self._row_col_to_state(row, col)
                    printstring += f'{self.states[state]}\t'
                printstring += '\n'
            print(printstring)
            return printstring
        elif mode == 'human':
            compimage = np.zeros((180*self.nrows, 180*self.ncols, 3), dtype=int)
            for row in range(self.nrows):
                rowslice = slice(180*row, 180*(row+1))
                for col in range(self.ncols):
                    colslice = slice(180*col, 180*(col+1))
                    state = self._row_col_to_state(row, col)
                    compimage[rowslice, colslice] = self._get_state_image(state).astype(int)
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.imshow(compimage)
            ax.set_xticks([])
            ax.set_yticks([])
            return compimage
        return None

    def _is_slippery_state(self, state):
        return self.states[state] == 's'

    def _slip_action(self, state, action):
        if action in (2, 3):
            return self.rng.choice([0, 1, action])
        elif action in (0, 1):
            return self.rng.choice([2, 3, action])

    def _next_state(self, state, action):
        if self.is_slippery and self._is_slippery_state(state):
            action = self._slip_action(state, action)

        row, col = self._state_to_row_col(state)
        if action == 0:
            row = max(row - 1, 0)
        elif action == 1:
            row = min(row + 1, self.nrows - 1)
        elif action == 2:
            col = max(col - 1, 0)
        elif action == 3:
            col = min(col + 1, self.ncols - 1)
        return self._row_col_to_state(row, col)

    def _state_to_row_col(self, state):
        row = int(np.floor(state / self.ncols))
        col = state - row*self.ncols
        return row, col

    def _state_to_reward(self, state):
        r_dict = self.reward_map[self.states[state]]
        return self.rng.choice(r_dict['r'], p=r_dict['p'])

    def _state_to_done(self, state):
        return self.done_map[self.states[state]]

    def _is_done(self, state):
        return self._state_to_done(state) or (self.steps >= self.max_steps)

    def _row_col_to_state(self, row, col):
        return row*self.ncols + col

    def _get_state_image(self, state):
        return mpimg.imread(self.image_map[self.states[state]])

    def _get_agent_image(self):
        return mpimg.imread(self.image_map['A'])


def animate_scenes(scenes):
    plt.close()
    fig, ax = plt.subplots()
    ax.set_xticks([])
    ax.set_yticks([])

    def animate(t):
        return (ax.imshow(scenes[t]),)

    anim = animation.FuncAnimation(fig, animate, frames=len(scenes), interval=500, blit=True)
    plt.close()
    return anim


def perform_action_sequence(actions, environment, render=False, seed=None):
    state = environment.reset(seed)
    steps, reward_acc = 0, 0
    done = False
    scenes = []
    anim = None
    
    for action in actions:
        if render:
            scenes += [environment.render()]
            plt.close()
        state, reward, done, info = environment.step(action)
        reward_acc += reward
        steps += 1

    if render:
        scenes += [environment.render()]
        anim = animate_scenes(scenes)
        
    environment.reset()
    return reward_acc, steps, anim


def play_episode(agent, environment, render=False, seed=None):
    state = environment.reset(seed)

    steps, reward_acc = 0, 0
    done = False
    scenes = []
    anim = None
    
    while not done:
        if render:
            scenes += [environment.render()]
            plt.close()
        action = agent.get_action(state, epsilon=0)
        state, reward, done, info = environment.step(action)
        reward_acc += reward
        steps += 1

    if render:
        scenes += [environment.render()]
        anim = animate_scenes(scenes)
        
    environment.reset()
    return reward_acc, steps, anim


def visualize_qvalues(agent, environment):
    envmap = environment.showmap()
    plt.close()
    fig = plt.figure()
    ax_val = fig.add_subplot(1, 2, 1)
    ax_val.set_xticks([])
    ax_val.set_yticks([])
    ax_action = fig.add_subplot(1, 2, 2)
    ax_action.set_xticks([])
    ax_action.set_yticks([])

    ax_val.imshow(envmap)
    ax_action.imshow(envmap)
    rowdim = 180
    coldim = 180
    edgesize = 40
    for row in range(environment.nrows):
        rowstart = rowdim*row
        rowstop = rowdim*(row+1)
        rowmid = rowstart + np.floor((rowstop-rowstart) / 2).astype(int)
        for col in range(environment.ncols):
            colstart = coldim*col
            colstop = coldim*(col+1)
            colmid = colstart + np.floor((colstop-colstart) / 2).astype(int)
            state = environment._row_col_to_state(row, col)
            if not environment._state_to_done(state):
                max_value = np.max(agent.qtable[state])
                ax_val.annotate(f'{max_value:.2f}', xy=(colmid-40, rowmid+20), color='r')
                action = agent.get_action(state, epsilon=0)
                if action == 0:
                    ax_action.arrow(colmid, rowstop - edgesize, 0, -(rowdim - 2*edgesize), width=1.0, color='r',
                                    head_width=20, head_length=20, length_includes_head=True)
                elif action == 1:
                    ax_action.arrow(colmid, rowstart + edgesize, 0, rowdim - 2*edgesize, width=1.0, color='r',
                                    head_width=20, head_length=20, length_includes_head=True)
                elif action == 2:
                    ax_action.arrow(colstop - edgesize, rowmid, -(coldim - 2*edgesize), 0, width=1.0, color='r',
                                    head_width=20, head_length=20, length_includes_head=True)
                elif action == 3:
                    ax_action.arrow(colstart + edgesize, rowmid, coldim - 2*edgesize, 0, width=1.0, color='r',
                                    head_width=20, head_length=20, length_includes_head=True)
    return fig
