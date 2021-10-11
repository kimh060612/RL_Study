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

    def update(self, state, action, reward, state_next):
        a_ = np.argmax(self.QFunction[state_next])
        self.QFunction[state, action] = self.QFunction[state, action] + self.lr * \
                                            (reward + self.Discount * self.QFunction[state_next, a_] - self.QFunction[state, action])    

if __name__ == "__main__":
    env = gym.make('FrozenLake-v1')
    agent = TDAgent(16, 4)
    total_reward_train = 0

    for epoch in range(EPISODES):
        state_ = env.reset()
        
        for t in range(1000):
            env.render()
            action = agent.get_action(state_)
            observation, reward, done, info = env.step(action)
            agent.update(state=state_, action=action, reward=reward, state_next=observation)
            state_ = observation
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
