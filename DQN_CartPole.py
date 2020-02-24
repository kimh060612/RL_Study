import sys
import tensorflow as tf
import gym
import numpy as np
import random
import pylab
from collections import deque
import keras.backend as K
from keras.optimizers import Adam
from keras.layers import Dense
from keras.models import Sequential

Num_Episode = 5000

class DQNAgnet:
    def __init__(self, StateNum, ActionNum, Load_need = False):
        self.size_state = StateNum
        self.size_action = ActionNum
        self.learning_rate = 0.001
        self.replay_memory = deque(maxlen=2000)

        # hyper-parameter in DQN
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.BatchSize = 64
        self.train_start = 1000

        # Off-policy learning means that there are two policies (Behavior, target)
        self.behavior_model = self.NeuralNet_Model()
        self.target_model = self.NeuralNet_Model()

        if Load_need:
            self.load_model("D:\\reinforcement-learning-kr\\2-cartpole\\1-dqn\\save_model\\cartpole_dqn_trained.h5")


    def DoAction(self, state): #Do Action following behavior policy
        if (np.random.rand() < self.epsilon):
            return random.randrange(self.size_action)
        else:
            Q_value = self.behavior_model.predict(state)
            return np.argmax(Q_value[0])


    def Update_Approximator(self):
        # Epsilon 
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        mini_batch = random.sample(self.replay_memory, self.BatchSize)

        states = np.zeros((self.BatchSize, self.size_state))
        next_states = np.zeros((self.BatchSize, self.size_state))
        actions, rewards, dones = [], [], []

        for i in range(self.BatchSize):
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]
            dones.append(mini_batch[i][4])

        target = self.behavior_model.predict(states)
        target_val = self.target_model.predict(next_states)

        for i in range(self.BatchSize):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.discount_factor * (np.amax(target_val[i]))

        self.behavior_model.fit(states, target, batch_size=self.BatchSize, epochs=1, verbose=0)

    def NeuralNet_Model(self): #Neural Network Function Approximation
        model = Sequential()
        model.add(Dense(24, input_dim=self.size_state, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(24, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.size_action, activation="linear", kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # Save <S, A, R, S`> for Replay memory & done parameter is to save the Episode is done. 'Cause there is difference in Q-function update mdethod.
    def Memory_save(self, state, action, reward, next_state, done):
        self.replay_memory.append((state, action, reward, next_state, done))

    def load_model(self, path = ""): # Get Q-function Approximator about Behavior Policy
        self.behavior_model.load_weights(path)

    def update_target_model(self):
        self.target_model.set_weights(self.behavior_model.get_weights())

if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgnet(state_size, action_size, False)
    scores = []
    episodes = []
    
    for episode in range(Num_Episode):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            env.render()

            action = agent.DoAction(state)
            next_state, reward, done, info = env.step(action) #Evironment gives 
            next_state = np.reshape(next_state, [1, state_size])
            reward = reward if not done or score == 499 else -100
            agent.Memory_save(state, action, reward, next_state, done)

            if len(agent.replay_memory) >= agent.train_start:
                agent.Update_Approximator()

            score += reward
            state = next_state

            if done:
                
                agent.update_target_model()
                score = score if score == 500 else score + 100

                scores.append(score)
                episodes.append(episode)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("D:\\reinforcement-learning-kr\\2-cartpole\\1-dqn\\save_graph\\cartpole_dqn.png")
                print("episode:", episode, "  score:", score, "  memory length:", len(agent.replay_memory), "  epsilon:", agent.epsilon)

                if np.mean(scores[-min(10, len(scores)):]) > 490:
                    agent.behavior_model.save_weights("D:\\RL_Study\\Model_save\\cartpole_dqn.h5")
                    sys.exit()


