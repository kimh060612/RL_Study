import numpy as np
import os
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

def PolicyEvaluation(Epoch):
    for epoch in range(Epoch):
        v = copy.deepcopy(ValueFunction)
        for i in range(STATE_HEIGHT):
            for j in range(STATE_WIDTH):
                val = 0
                if i == j and (i == 0 or i == 4):
                    val = 0
                else :
                    for act in action:
                        i_, j_ = get_state([i, j], act)
                        val += PolicyFunction[act][i][j] * (Env[i_][j_] + Discount * v[i_][j_])
                ValueFunction[i][j] = val
        delta = np.max(np.array([np.max(np.abs(v[i] - ValueFunction[i])) for i in range(len(ValueFunction))]))
        if (epoch + 1) % 10 == 0:
            print("Epoch", epoch + 1)
            print(ValueFunction)
        if delta < EPS or delta > INF:
            break

def PolicyImprovement(Epoch):
    
    isStable = True

    for epoch in range(Epoch):
        pi = copy.deepcopy(PolicyFunction)
        Q_pi = np.zeros((4, 5, 5))
        for act in action:
            for i in range(STATE_HEIGHT):
                for j in range(STATE_WIDTH):
                    if i == j and (i == 0 or i == 4):
                        Q_pi[act][i][j] = 0
                    else : 
                        i_, j_ = get_state([i, j], act)
                        Q_pi[act][i][j] = Env[i_][j_] + Discount * ValueFunction[i_][j_]
        
        for i in range(STATE_HEIGHT):
            for j in range(STATE_WIDTH):
                a_pi = 0
                for act in action:
                    if Q_pi[a_pi][i][j] < Q_pi[act][i][j]:
                        a_pi = act
                count = 0
                for act in action:
                    if Q_pi[a_pi][i][j] == Q_pi[act][i][j]:
                        count += 1
                PolicyFunction[a_pi][i][j] = 1 / count
                for act in action:
                    if not act == a_pi:
                        PolicyFunction[act][i][j] = 0
        
        if not np.all(pi == PolicyFunction):
            isStable = False
        if (epoch + 1) % 10 == 0:
            print("Epoch: ", epoch + 1)
            action_table = []
            for i in range(STATE_HEIGHT):
                tmp = []
                for j in range(STATE_WIDTH):
                    idx = np.argmax(PolicyFunction[:,i,j])
                    tmp.append(action[idx])
                action_table.append(tmp)
            print(np.array(action_table))
        if isStable:
            break

if __name__ == "__main__":
    PolicyFunctionInitialization()

    for _ in range(100):
        PolicyEvaluation(100)
        PolicyImprovement(100)
