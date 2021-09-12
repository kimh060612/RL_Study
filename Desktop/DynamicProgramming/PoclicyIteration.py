import numpy as np
import os
import copy

EPS = 1e-10
Discount = 0.9
Reward = -1
Env = np.zeros((5, 5))
Env[2][2], Env[3][3] = 10, -5
ValueFunction = np.zeros((5, 5))
QFunc = np.zeros((5, 5, 4))
ProbabilityTransition = np.random.rand(4, 5, 5)
Policy = np.random.randint(0, 5, size=(5, 5))
action_grid = [(-1, 0), (1, 0), (0, -1), (0, 1)]
action = [0, 1, 2, 3]

def get_state(state, action):
    
    state[0] += action_grid[action][0]
    state[1] += action_grid[action][1]
    
    if state[0] < 0 :
        state[0] = 0
    elif state[0] > 3 :
        state[0] = 3
    
    if state[1] < 0 :
        state[1] = 0
    elif state[1] > 3 :
        state[1] = 3
    
    return state[0], state[1]

def DefineProbabilityTransition():
    for i in range(5):
        for j in range(5):
            for k in range(4):
                if i == j and (i == 0 or i == 4):
                    ProbabilityTransition[k][i][j] = 0.00
                else :
                    ProbabilityTransition[k][i][j] = 0.25

def PolicyEvaluation():

    for epoch in range(1000):
        v = copy.deepcopy(ValueFunction)
        for i in range(5):
            for j in range(5):
                val = 0
                if i == j and ((i == 0) or (i == 4)):
                    ValueFunction[i][j] = val
                else :
                    for act in action:
                        i_, j_ = get_state([i, j], act)
                        val += ProbabilityTransition[act][i_][j_] * (Env[i_][j_] + Discount * ValueFunction[i_][j_])
                    ValueFunction[i][j] = val
        delta = np.max(np.max(np.abs(v - ValueFunction), axis=1).reshape(1, -1))
        if epoch % 10 == 0:
            print("Epoch: " + str(epoch))
            print(ValueFunction)
        if delta < EPS:
            break

def PolicyImprovement():
    
    IsStable = True

    for epoch in range(1000):
        for i in range(5):
            for j in range(5):
                if i == j and (i == 0 or i == 4):
                    Policy[i][j] = -1
                else :
                    val = 0
                    v_pi = []
                    for k in range(4):
                        i_, j_ = get_state([i, j], k)
                        val += ProbabilityTransition[k][i_][j_] * (Env[i_][j_] + Discount * ValueFunction[i_][j_])
                     
                    pass
        
        if epoch % 10 == 0:
            print("Epoch: " + str(epoch))
            print(ValueFunction)

if __name__ == "__main__":
    state = (0, 0)

    DefineProbabilityTransition()
    PolicyEvaluation()
