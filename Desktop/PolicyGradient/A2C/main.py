import gym
import numpy as np
import tensorflow as tf
from agent import agent
import json
import matplotlib.pyplot as plt

np.random.seed(0)
tf.enable_eager_execution()

EPISODES=1000
DISCOUNT = 0.99

if __name__ == "__main__":
    env = gym.make('MountainCar-v0')
    with open("./config.json", "r") as file:
        config = json.load(file)
    Agent = agent(state_space=4, action_space=3, N_batch=1, discount=config["discount"], LR=7e-4, step=1)
    
    reward_plot = []

    for i_episode in range(EPISODES):
        state_now = env.reset()
        total_reward_train = 0

        for t in range(1000):
            env.render()
            action = Agent.get_action(state_now)
            observation, reward, done, _ = env.step(action)
            Agent.save_batch(state=state_now, action=action, reward=reward, n_state=observation)
            state_now = observation
            total_reward_train += reward
            if done:
                print("Episode {} finished after {} timesteps in reward {}".format(i_episode + 1, t + 1, total_reward_train))
                reward_plot.append(total_reward_train)
                break
    
    print(total_reward_train / EPISODES)
    plt.plot(reward_plot)
    plt.show()