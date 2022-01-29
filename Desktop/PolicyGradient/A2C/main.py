import gym
import numpy as np
import tensorflow as tf
from agent import agent
import json
import matplotlib.pyplot as plt

np.random.seed(0)
# tf.enable_eager_execution()

EPISODES=1000
DISCOUNT = 0.99

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    with open("./config.json", "r") as file:
        config = json.load(file)
    Agent = agent(state_space=env.observation_space.shape[0], action_space=env.action_space.n, N_batch=1, discount=config["discount"], LR=1e-3, step=1)
    
    reward_plot = []

    for i_episode in range(EPISODES):
        state_now = env.reset()
        total_reward_train = 0
        done = False
        t = 1
        while not done:
            env.render()
            action = Agent.get_action(state_now)
            observation, reward, done, _ = env.step(action)
            Agent.save_batch(state=state_now, action=action, reward=reward, n_state=observation, done=done)
            state_now = observation
            total_reward_train += reward
            t += 1
            if done:
                print("Episode {} finished after {} timesteps in reward {}".format(i_episode + 1, t + 1, total_reward_train))
                reward_plot.append(total_reward_train)
                break
    
    print("Mean Train Reward: ", sum(reward_plot) / len(reward_plot))
    plt.plot(reward_plot)
    plt.show()