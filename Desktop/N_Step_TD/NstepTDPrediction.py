import gym
import numpy as np

GAMMA=0.99
ALPHA = 0.1
EPISODES=100000
N = 5

class TDAgent:
    def __init__(self, ObserveSpace):
        self.space = ObserveSpace
        self.ValueFunction = np.zeros(ObserveSpace)
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
        if not done:
            G += pow(self.Discount, N) * self.ValueFunction[state_next]
        s_ = self.memory_state[self.memory_length() - N]
        self.ValueFunction[s_] = self.ValueFunction[s_] + self.lr * (G - self.ValueFunction[s_])

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
    agent = TDAgent(16)

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
    print(agent.ValueFunction.reshape(4, 4))