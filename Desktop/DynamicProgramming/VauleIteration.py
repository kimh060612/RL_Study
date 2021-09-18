import numpy as np
import copy

EPS = 1e-10
INF = 1e18
Discount = 0.9
Env = np.ones((5, 5)) * (-1)
Env[2][2], Env[3][3] = 10, -5
STATE_WIDTH = 5
STATE_HEIGHT = 5

ValueFunction = np.zeros((5, 5))
QFunc = np.zeros((5, 5, 4))
PolicyFunction = np.random.rand(4, 5, 5)
action_grid = [(-1, 0), (1, 0), (0, -1), (0, 1)]
action = [0, 1, 2, 3]

def get_state(state, action):
    
    action_grid = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    state[0]+=action_grid[action][0]
    state[1]+=action_grid[action][1]
    
    if state[0] < 0 :
        state[0] = 0
    elif state[0] > 3 :
        state[0] = 3
    
    if state[1] < 0 :
        state[1] = 0
    elif state[1] > 3 :
        state[1] = 3
    
    return state[0], state[1]

def PolicyFunctionInitialization():
    for i in range(STATE_HEIGHT):
        for j in range(STATE_WIDTH):
            for act in action:
                if i == j and (i == 0 or i == 4):
                    PolicyFunction[act][i][j] = -1
                else :
                    PolicyFunction[act][i][j] = 0.25
    PolicyFunction[:,0,0] = [0] * 4
    PolicyFunction[:,4,4] = [0] * 4

def ValueIteration(Epoch):

    for epoch in range(Epoch):
        value = copy.deepcopy(ValueFunction)
        for i in range(STATE_HEIGHT):
            for j in range(STATE_WIDTH):
                if i == j and (i == 0 and i == 4):
                    ValueFunction[i][j] = 0
                else :
                    max_val = -987654321
                    for act in action:
                        i_, j_ = get_state([i, j], act)
                        val = (Env[i_][j_] + Discount * ValueFunction[i_][j_])
                        if max_val <= val:
                            max_val = val
                    ValueFunction[i][j] = max_val
        delta = np.max(np.array([np.max(np.abs(value[i] - ValueFunction[i])) for i in range(len(ValueFunction))]))            
        if delta < EPS:
            break
        if (epoch + 1) % 10:
            print("Epoch: ", epoch + 1)
            print(ValueFunction)
              
if __name__ == "__main__":

    PolicyFunctionInitialization()
    ValueIteration(500)
    
    Action_policy = []

    for i in range(STATE_HEIGHT):
        tmp = []
        for j in range(STATE_WIDTH):
            max_action = 0
            max_val = -987654321
            for act in action:
                i_, j_ = get_state([i, j], act)
                val = (Env[i_][j_] + Discount * ValueFunction[i_][j_])
                if max_val < val:
                    max_action = act
                    max_val = val
            tmp.append(max_action)
        Action_policy.append(tmp)
    print(np.array(Action_policy))