import gym
import numpy as np
import random

GAMMA=0.9
EPISODES=1000

class MCAgent:
    def __init__(self, ObserveSpace):
        self.space = ObserveSpace
        self.ValueFunction = np.random.rand(ObserveSpace)
        self.state_count = np.zeros(ObserveSpace)
        self.memory_state = [] # store (state, reward) tuple
        self.memory_reward = []
        self.G = np.zeros(ObserveSpace)
        self.Discount = GAMMA
    
    def update(self):
        self.memory_state.reverse()
        self.memory_reward.reverse()

        for t in range(len(self.memory_state)):
            s = self.memory_state[t]
            prev = self.memory_state[t + 1:]
            if not s in prev:
                self.G = self.Discount * self.G + self.memory_reward[t]
                self.state_count[s] += 1
                self.ValueFunction[s] = self.ValueFunction[s] + (1 / self.state_count[s]) * (self.G - self.ValueFunction[s])

    def memorize(self, observation, reward):
        self.memory_state.append(observation)
        self.memory_reward.append(reward)
    
    def initialize(self):
        self.memory_reward = []
        self.memory_state = []
        self.G = 0 

if __name__ == "__main__":
    
    env = gym.make('FrozenLake-v1')
    agent = MCAgent(16)

    for i_episode in range(EPISODES):
        observation = env.reset()
        agent.initialize()

        for t in range(1000):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            agent.memorize(observation=observation, reward=reward)    
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                agent.update()
                break
    env.close() 

    print("-------------Value Function-----------------")
    print(agent.ValueFunction.reshape(4, 4))