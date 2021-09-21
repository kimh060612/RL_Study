import gym
import numpy as np

GAMMA=0.99
EPISODES=100000
TEST_EPISODE = 100
EPS = 0.1

class MCAgent:
    def __init__(self, ObserveSpace, actionSpace):
        self.space = ObserveSpace
        self.QFunction = np.random.rand(ObserveSpace, actionSpace)
        self.state_count = np.zeros((ObserveSpace, actionSpace))
        self.memory_sa = [] # store (state, action) tuple
        self.memory_reward = []
        self.policy = np.random.randint(0, 4, size=(ObserveSpace, ))
        self.Discount = GAMMA
        self.eps = EPS
    
    def get_action(self, state):
        if np.random.rand() < self.eps:
            return np.random.randint(0, 4)
        else :
            return self.policy[state]

    def update(self):
        self.memory_sa.reverse()
        self.memory_reward.reverse()
        G = self.memory_reward[0]

        for t in range(1, len(self.memory_sa)):
            s, a = self.memory_sa[t]
            prev = self.memory_sa[t + 1:]
            G = self.Discount * G + self.memory_reward[t]
            if not (s, a) in prev:
                self.state_count[s][a] += 1
                self.QFunction[s][a] = self.QFunction[s][a] + (1 / self.state_count[s][a]) * (G - self.QFunction[s][a]) # (1 / self.state_count[s][a])  
                self.policy[s] = np.argmax(self.QFunction[s])

    def memorize(self, observation, action, reward):
        self.memory_sa.append((observation, action))
        self.memory_reward.append(reward)
    
    def initialize(self):
        self.memory_reward = []
        self.memory_sa = []

if __name__ == "__main__":
    
    env = gym.make('FrozenLake-v1')
    agent = MCAgent(16, 4)

    for i_episode in range(EPISODES):
        observation = env.reset()
        agent.initialize()
        state_now = 0

        for t in range(1000):
            env.render()
            action = agent.get_action(state=state_now)
            observation, reward, done, _ = env.step(action)
            agent.memorize(observation=state_now, action=action, reward=reward)
            state_now = observation

            if done:
                print("Episode finished after {} timesteps".format(t+1))
                agent.update()
                break
    
    total_reward_test = 0
    for i_episode in range(TEST_EPISODE):
        observation = env.reset()
        state_now = 0

        for t in range(1000):
            env.render()
            action = agent.get_action(state=state_now)
            observation, reward, done, _ = env.step(action)
            total_reward_test += reward
            state_now = observation

            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
    print(total_reward_test)
    env.close() 

    print("-------------Policy-----------------")
    print(agent.policy.reshape(4, 4))
    print(agent.QFunction)