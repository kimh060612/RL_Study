import gym
import numpy as np

GAMMA=0.99
ALPHA = 0.1
EPISODES=100000
N = 5
DIM = 2
WIDTH = 4
HEIGHT = 4

def polynomial(state, CMatrix):
    y = state / HEIGHT
    x = state % WIDTH

    feature = [ (x ** CMatrix[i][0]) * (y ** CMatrix[i][1]) for i in range(DIM) ]
    feature = np.array(feature)
    return np.transpose(feature)    

class TDAgent:
    def __init__(self, ObserveSpace, dim):
        self.space = ObserveSpace
        self.weight = np.zeros(dim)
        self.Constant = np.random.randint(0, dim, size=(dim, 2))
        self.memory_state = [] # store (state, reward) tuple
        self.memory_reward = []
        self.Discount = GAMMA
        self.lr = ALPHA
    
    def update(self, state_next, done):
        if self.memory_length() == 0:
            return
        G = 0
        for t in range(N):
            G = self.Discount * G + self.memory_reward[self.memory_length() - 1 - t]
        
        s_ = self.memory_state[self.memory_length() - N]
        X = polynomial(s_, self.Constant)
        value = np.matmul(self.weight, X)
        if not done:
            G += pow(self.Discount, N) * value
        self.weight = self.weight + self.lr * (G - value) * X

    def memorize(self, observation, reward):
        self.memory_state.append(observation)
        self.memory_reward.append(reward)
    
    def memory_length(self):
        return len(self.memory_state)

    def initialize(self):
        self.memory_reward = []
        self.memory_state = []

if __name__ == "__main__":
    
    env = gym.make('FrozenLake-v1')
    agent = TDAgent(16, DIM)

    for i_episode in range(EPISODES):
        state_now = env.reset()
        agent.initialize()

        for t in range(1000):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, _ = env.step(action)
            agent.memorize(observation=state_now, reward=reward)
            state_now = observation

            if t - N + 1 >= 0:
                agent.update(observation, done)

            if done:
                print("Episode finished after {} timesteps".format(t+1))    
                break
    env.close() 

    print("-------------Value Function-----------------")
    X_matrix = [ np.transpose(polynomial(s_, agent.Constant)) for s_ in range(16) ]
    X_matrix = np.array(X_matrix).T
    print(np.matmul(agent.weight, X_matrix).reshape(4, 4)) # .
