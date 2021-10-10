import numpy as np
import gym
import matplotlib.pyplot as plt

GAMMA=0.99
EPISODES=100000
ALPHA = 0.1
EPSILLON = 0.1
TEST_EPISODE = 1000

class TDAgent:
    def __init__(self, observe_space, action_space):
        self.observe_space = observe_space
        self.action_space = action_space
        self.Q1Function = np.random.rand(self.observe_space, self.action_space)
        self.Q2Function = np.random.rand(self.observe_space, self.action_space)
        self.lr = ALPHA
        self.Discount = GAMMA
        self.EPS = EPSILLON

    def get_action(self, state):
        if np.random.rand() < self.EPS:
            return np.random.randint(0, 4)
        else : 
            return np.argmax(self.Q1Function[state, :] + self.Q2Function[state, :])

    def update(self, state, action, reward, state_next):
        if np.random.rand() < 0.5 :
            a_ = np.argmax(self.Q2Function[state_next])
            self.Q1Function[state, action] += self.lr * (reward + self.Discount * self.Q2Function[state_next, a_] - self.Q1Function[state, action])
        else :
            a_ = np.argmax(self.Q1Function[state_next])
            self.Q2Function[state, action] += self.lr * (reward + self.Discount * self.Q1Function[state_next, a_] - self.Q2Function[state, action])

if __name__ == "__main__":
    env = gym.make('FrozenLake-v1')
    agent = TDAgent(16, 4)
    train_reward_list = []

    for epoch in range(EPISODES):
        state_ = env.reset()
        total_reward_train = 0
        t = 0
        while True:
            env.render()
            action = agent.get_action(state_)
            observation, reward, done, info = env.step(action)
            agent.update(state=state_, action=action, reward=reward, state_next=observation)
            state_ = observation
            t += 1
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                total_reward_train += reward
                break
        train_reward_list.append(total_reward_train)
    
    total_reward_test = 0
    for i_episode in range(TEST_EPISODE):
        state_now = env.reset()

        for t in range(1000):
            env.render()
            action = np.argmax(agent.Q1Function[state_now, :] + agent.Q2Function[state_now, :]) # agent.get_action(state=state_now)
            observation, reward, done, _ = env.step(action)
            state_now = observation

            if done:
                print("Episode finished after {} timesteps".format(t+1))
                total_reward_test += reward
                break
    
    print(sum(train_reward_list) / EPISODES)
    print(total_reward_test / TEST_EPISODE)
    env.close() 
    
    print(np.argmax(agent.Q1Function + agent.Q2Function, axis=1).reshape(4, 4))
    plt.plot(train_reward_list)
    plt.show()