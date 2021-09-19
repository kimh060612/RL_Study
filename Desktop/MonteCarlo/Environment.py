import numpy as np

class ENV:
    def __init__(self, STATE_HEIGHT, STATE_WIDTH, reward_pos = (3, 3), trap_pos = (2, 2)):
        self.height = STATE_HEIGHT
        self.width = STATE_WIDTH
        self.Env = np.ones((self.height, self.width)) * (-1)
        self.Env[reward_pos[0], reward_pos[1]] = 10
        self.Env[trap_pos[0], trap_pos[1]] = -10
        self.action_set = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.action_idx = [0, 1, 2, 3]
        
    def get_state(self, state, action):
        
        pass

    def get_action(self):
        pass

    