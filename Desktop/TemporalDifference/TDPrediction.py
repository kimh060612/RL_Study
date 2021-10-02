import numpy as np
import gym

GAMMA=0.9
EPISODES=10000
ALPHA = 0.5

class TDAgent:
    def __init__(self, observe_space, action_space):
        self.observe_space = observe_space
        self.ValueFunction = np.random.rand(self.observe_space)
        self.ValueFunction[self.observe_space - 1] = 0
        self.policy = np.random.randint(0, action_space, self.observe_space)
        self.lr = ALPHA
        self.Discount = GAMMA

    def get_action(self, state):
        return self.policy[state]

    def update(self, state, state_next, reward):
        self.ValueFunction[state] = self.ValueFunction[state] + self.lr * \
                                (reward + self.Discount * self.ValueFunction[state_next] - self.ValueFunction[state])

if __name__ == "__main__":
    env = gym.make('FrozenLake-v1')
    agent = TDAgent(16, 4)

    for epoch in range(EPISODES):
        observation = env.reset()
        state_ = 0
        for t in range(1000):
            env.render()
            action = agent.get_action(state_)
            observation, reward, done, info = env.step(action)
            agent.update(state=state_, state_next=observation, reward=reward)
            state_ = observation

            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
    env.close() 
    
    print(agent.ValueFunction.reshape((4, 4)))
    print(agent.policy.reshape((4, 4)))