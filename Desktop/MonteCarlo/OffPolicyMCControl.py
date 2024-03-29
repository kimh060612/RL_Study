import gym
import numpy as np

GAMMA=0.99
EPISODES=100000
TEST_EPISODE = 1000
EPS = 0.1

class MCAgent:
    def __init__(self, ObserveSpace, actionSpace):
        self.space = ObserveSpace
        self.action_space = actionSpace
        self.QFunction = np.random.rand(ObserveSpace, actionSpace) * 100
        self.C = np.zeros((ObserveSpace, actionSpace))
        self.behaviorPolicyFunction = np.random.rand(ObserveSpace, actionSpace)
        self.memory_sa = [] # store (state, action) tuple
        self.memory_reward = []
        self.policy = np.argmax(self.QFunction, axis=1)
        self.Discount = GAMMA
        self.eps = EPS
    
    def get_action(self, state):
        if np.random.rand() < self.eps:
            return np.random.randint(0, self.action_space)
        else :
            return np.argmax(self.behaviorPolicyFunction[state])
    
    def get_action_pi(self, state):
        return self.policy[state]

    def update(self):
        self.memory_sa.reverse()
        self.memory_reward.reverse()
        G = 0
        W = 1

        for t in range(len(self.memory_sa)):
            s, a = self.memory_sa[t]
            G = self.Discount * G + self.memory_reward[t]
            self.C[s][a] += W
            self.QFunction[s][a] = self.QFunction[s][a] + (W / self.C[s][a]) * (G - self.QFunction[s][a])
            self.policy[s] = np.argmax(self.QFunction[s])
            if not a == self.policy[s] :
                break
            W *= 1.0 / self.behaviorPolicyFunction[s][a]

    def memorize(self, observation, action, reward):
        self.memory_sa.append((observation, action))
        self.memory_reward.append(reward)
    
    def initialize(self):
        self.memory_reward = []
        self.memory_sa = []
        self.behaviorPolicyFunction = np.random.rand(self.space, self.action_space)

if __name__ == "__main__":
    
    env = gym.make('FrozenLake-v1')
    agent = MCAgent(16, 4)
    total_reward_train = 0
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
                total_reward_train += reward
                break
    
    total_reward_test = 0
    for i_episode in range(TEST_EPISODE):
        observation = env.reset()
        state_now = 0

        for t in range(1000):
            env.render()
            action = agent.get_action_pi(state=state_now)
            observation, reward, done, _ = env.step(action)
            total_reward_test += reward
            state_now = observation

            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
    
    print(total_reward_train / EPISODES)
    print(total_reward_test / TEST_EPISODE)
    env.close() 

    print("-------------Policy-----------------")
    print(agent.policy.reshape(4, 4))
    print(np.argmax(agent.QFunction, axis=1).reshape(4, 4))