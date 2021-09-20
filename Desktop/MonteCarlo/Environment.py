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
        
        x, y = state
        
        y += self.action_set[action][0]
        x += self.action_set[action][1]

        if x < 0 :
            x = 0
        elif x > (self.width - 1) :
            x = (self.width - 1)

        if y < 0 :
            y = 0
        elif y > (self.height - 1) :
            y = (self.height - 1)
        
        return [x, y], self.Env[y][x]

class MC_agent:
    def __init__(self):
        self.action_grid = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.action_text= ['U', 'D', 'L', 'R']
        self.grid_width = 5
        self.grid_height = self.grid_width
        self.value_table = np.zeros((self.grid_width, self.grid_height))
        self.e = .1
        self.learning_rate = .01
        self.discount_factor = .95
        self.memory=[]
    
    