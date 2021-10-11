import gym
import numpy as np

GAMMA=0.99
EPISODES=100000
TEST_EPISODE = 1000
EPS = 0.1
ALPHA = 0.1
N = 2

class TDAgent:
    def __init__(self, ObserveSpace, actionSpace):
        self.space = ObserveSpace
        self.action_space = actionSpace
        self.QFunction = np.random.rand(ObserveSpace, actionSpace)
        self.memory_sa = [] # store (state, action) tuple
        self.memory_reward = []
        self.Discount = GAMMA
        self.eps = EPS
        self.lr = ALPHA
    
    def get_action(self, state):
        if np.random.rand() < self.eps:
            return np.random.randint(0, self.action_space)
        else :
            return np.argmax(self.QFunction[state])

    def update(self, s_next, a_next, done):
        G = 0

        for t in range(N):
            G = self.Discount * G + self.memory_reward[len(self.memory_reward) - 1 - t]
        
        if not done:
            G += pow(self.Discount, N) * self.QFunction[s_next][a_next]
        s, a = self.memory_sa[len(self.memory_sa) - 1 - N]
        self.QFunction[s][a] = self.QFunction[s][a] + self.lr * (G - self.QFunction[s][a])   

    def memorize(self, observation, action, reward):
        self.memory_sa.append((observation, action))
        self.memory_reward.append(reward)
    
    def memory_length(self):
        return len(self.memory_reward)

    def initialize(self):
        self.memory_reward = []
        self.memory_sa = []

if __name__ == "__main__":
    
    env = gym.make('FrozenLake-v1')
    agent = TDAgent(16, 4)

    total_reward_train = 0
    for i_episode in range(EPISODES):
        state_now = env.reset()
        agent.initialize()
        T = 987654321
        action = agent.get_action(state=state_now)
        for t in range(1000):
            env.render()
            observation, reward, done, _ = env.step(action)
            
            if done:
                T = t + 1
            else :
                action_next = agent.get_action(observation)

            agent.memorize(observation=state_now, action=action, reward=reward)
            if t - N + 1 >= 0:
                agent.update(observation, action_next, done)

            state_now = observation
            action = action_next
            
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                total_reward_train += reward
                break
    
    total_reward_test = 0
    for i_episode in range(TEST_EPISODE):
        state_now = env.reset()

        for t in range(1000):
            env.render()
            action = np.argmax(agent.QFunction[state_now]) # agent.get_action(state=state_now)
            observation, reward, done, _ = env.step(action)
            state_now = observation

            if done:
                print("Episode finished after {} timesteps".format(t+1))
                total_reward_test += reward
                break
    
    print(total_reward_train / EPISODES)
    print(total_reward_test / TEST_EPISODE)
    env.close() 

    print(np.argmax(agent.QFunction, axis=1).reshape(4, 4))