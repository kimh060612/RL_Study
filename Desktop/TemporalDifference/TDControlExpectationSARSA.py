import numpy as np
import gym

GAMMA=0.9
EPISODES=100000
ALPHA = 0.1
EPSILLON = 0.1
TEST_EPISODE = 1000

class TDAgent:
    def __init__(self, observe_space, action_space):
        self.observe_space = observe_space
        self.action_space = action_space
        self.QFunction = np.random.rand(self.observe_space, self.action_space)
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
        mean_ = (1 - self.EPS) * self.QFunction[state_next, a_]
        mean_ += np.mean(self.QFunction[state_next]) * self.EPS 
        self.QFunction[state, action] = self.QFunction[state, action] + self.lr * \
                                            (reward + self.Discount * mean_ - self.QFunction[state, action])    

if __name__ == "__main__":
    env = gym.make('FrozenLake-v1')
    agent = TDAgent(16, 4)
    total_reward_train = 0

    for epoch in range(EPISODES):
        observation = env.reset()
        state_ = 0
        
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
    
    print(total_reward_train / EPISODES)
    print(total_reward_test / TEST_EPISODE)
    env.close() 
    
    print(np.argmax(agent.QFunction, axis=1).reshape(4, 4))
