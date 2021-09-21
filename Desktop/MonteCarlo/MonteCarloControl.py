import gym
import numpy as np

GAMMA=0.9
EPISODES=10000

class MCAgent:
    def __init__(self, ObserveSpace, actionSpace):
        self.space = ObserveSpace
        self.QFunction = np.random.rand(ObserveSpace, actionSpace)
        self.state_count = np.zeros((ObserveSpace, actionSpace))
        self.memory_sa = [] # store (state, action) tuple
        self.memory_reward = []
        self.policy = np.random.randint(0, 4, size=(ObserveSpace, ))
        self.Discount = GAMMA
    
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
                self.QFunction[s][a] = self.QFunction[s][a] + (1 / self.state_count[s][a]) * (G - self.QFunction[s][a]) 
                self.policy[s] = np.argmax(self.QFunction[s])

    def memorize(self, observation, action, reward):
        self.memory_sa.append((observation, action))
        self.memory_reward.append(reward)
    
    def initialize(self):
        self.memory_reward = []
        self.memory_sa = []

if __name__ == "__main__":
    
    # This Frozen Lake Environment Cannot be solved by Simple MC Control. Cause it doesn't statisfy the ES (assumption of Coverage)
    env = gym.make('FrozenLake-v1')
    agent = MCAgent(16, 4)

    for i_episode in range(EPISODES):
        observation = env.reset()
        agent.initialize()
        state_now = 0

        for t in range(10000):
            env.render()
            action = agent.policy[state_now]
            observation, reward, done, _ = env.step(action)
            agent.memorize(observation=state_now, action=action, reward=reward)
            state_now = observation

            if done:
                print("Episode finished after {} timesteps".format(t+1))
                agent.update()
                break
    env.close() 

    print("-------------Policy-----------------")
    print(agent.policy.reshape(4, 4))
    print(agent.QFunction)