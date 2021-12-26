import gym
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
tf.random.set_seed(0)

EPISODES=1000
DISCOUNT = 0.99
LR = 1e-3
optimizer = tf.keras.optimizers.Adam(learning_rate=LR)

def policyDefinition(input_size, n_actions):
    
    model = keras.Sequential()
    model.add(keras.layers.Dense(128, activation='relu', kernel_initializer='random_normal', input_dim=input_size))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(64, kernel_initializer='random_normal', activation='relu'))
    model.add(keras.layers.Dense(n_actions, kernel_initializer='random_normal', activation='softmax'))
    
    return model

class REINFORCEAgent:
    def __init__(self, state_space, action_space):
        self.MemoryList = []
        self.policy = policyDefinition(state_space, action_space)
        # self.policy.summary()

    def get_action(self, state):
        state = tf.constant(np.expand_dims(state, axis=0))
        action_prob = self.policy(state)
        action = tf.argmax(action_prob[0])
        return action.numpy()

    def update(self):
        with tf.GradientTape() as tape:
            loss = 0
            for t in range(0, len(self.MemoryList) - 1):
                G = 0
                s, a, r_t1 = self.MemoryList[t]
                for t_ in range(t + 1, len(self.MemoryList)): 
                    _, _, r_t2 = self.MemoryList[t_]
                    G += pow(DISCOUNT, t_ - t) * r_t1
                    r_t1 = r_t2
                action_prob = self.policy(np.expand_dims(s, axis=0))
                action_pi = action_prob[0, a]
                loss += pow(DISCOUNT, t) * G * tf.math.log(action_pi)
            loss = -1 * loss
            print(loss)
            grad = tape.gradient(loss, self.policy.trainable_variables)
            optimizer.apply_gradients(zip(grad, self.policy.trainable_variables))

    def memory(self, state, action, reward):
        self.MemoryList.append((state, action, reward))
    
    def initialize(self):
        self.MemoryList = []

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    # agent = REINFORCEAgent(2, 3)
    agent = REINFORCEAgent(4, 2)

    reward_plot = []

    for i_episode in range(EPISODES):
        state_now = env.reset()
        total_reward_train = 0

        for t in range(1000):
            env.render()
            action = agent.get_action(state_now)
            observation, reward, done, _ = env.step(action)
            agent.memory(state=state_now, action=action, reward=reward)
            state_now = observation
            total_reward_train += reward
            if done:
                print("Episode {} finished after {} timesteps in reward {}".format(i_episode + 1, t + 1, total_reward_train))
                agent.update()
                agent.initialize()
                reward_plot.append(total_reward_train)
                break
    
    print(total_reward_train / EPISODES)
    plt.plot(reward_plot)
    plt.show()