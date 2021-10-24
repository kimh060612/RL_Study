import numpy as np
import gym

GAMMA=0.99
EPISODES=100000
ALPHA = 0.1
EPSILLON = 0.1
TEST_EPISODE = 1000

class TDAgent:
    def __init__(self, observe_space, action_space):
        self.observe_space = observe_space
        self.action_space = action_space
        self.QFunction = np.zeros((self.observe_space, self.action_space))
        self.lr = ALPHA
        self.Discount = GAMMA
        self.EPS = EPSILLON

    def get_action(self, state):
        if np.random.rand() < self.EPS:
            return np.random.randint(0, 4)
        else : 
            return np.argmax(self.QFunction[state])

    def update(self, state, action, reward, state_next, action_next):
        self.QFunction[state, action] = self.QFunction[state, action] + self.lr * \
                                            (reward + self.Discount * self.QFunction[state_next, action_next] - self.QFunction[state, action])    

if __name__ == "__main__":
    env = gym.make('FrozenLake-v1')
    agent = TDAgent(16, 4)
    total_reward_train = 0

    for epoch in range(EPISODES):
        state_ = env.reset()
        action = agent.get_action(state_)
        for t in range(1000):
            env.render()
            observation, reward, done, info = env.step(action)
            action_ = agent.get_action(observation)
            agent.update(state=state_, action=action, reward=reward, state_next=observation, action_next=action_)
            state_ = observation
            action = action_
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                total_reward_train += reward
                break

    total_reward_test = 0
    for i_episode in range(TEST_EPISODE):
        observation = env.reset()
        state_now = 0

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
    
    
