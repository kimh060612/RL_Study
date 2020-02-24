import sys
import numpy as np
import pylab
import tensorflow as tf
import random
import keras.backend as K
from keras.layers import Dense
from keras.models import Sequential
from collections import deque
from keras.optimizers import Adam
import gym

Num_Episode = 1000

class DQNAgent:
    def __init__(self, stateNum, actionNum, IsLoadWeight):
        self.state_size = stateNum
        self.action_size = actionNum
        self.ISload_model = IsLoadWeight
        self.path = "D:\\RL_Study\\Model_save\\cartpole_dqn_trained.h5"
        
        # Space for Experience replay
        self.Replay_memory = deque(maxlen=2000)

        #Hyper-parameters of DQN
        self.learning_rate = 0.003
        self.epsilon = 1
        self.epsilon_decay = 0.999
        self.min_epsilon = 0.01
        self.discount_rate = 0.99
        self.Train_start_Replay_memery_size = 1000
        self.BatchSize = 100

        # 2 Model in DQN, Behavior Policy, Target Policy
        # Q-funtion Approximation
        self.Behavior_model = self.Build_NN()
        self.Target_model = self.Build_NN()

        # Load Model
        if self.ISload_model :
            self.load_model(self.path)

    def DoPolicyAction(self,state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        else:
            Q_val = self.Behavior_model.predict(state)
            return np.argmax(Q_val[0])

    def Build_NN(self):
        Model = Sequential()
        Model.add(Dense(24, input_dim = self.state_size, activation="relu"))
        Model.add(Dense(24, activation="relu"))
        Model.add(Dense(self.action_size, activation="linear"))
        Model.summary()
        Model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))
        return Model

    # Saving <s, a, r, s`>  tuple to train the Approximator & save whether Episode is done. 
    def Save_replay_memory(self, Nowstate, action, reward, Nextstate, done):
        self.Replay_memory.append((Nowstate, action, reward, Nextstate, done))

    def Train_Model(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

        mini_batch_set = random.sample(self.Replay_memory, self.BatchSize)
        Nowstates = np.zeros((self.BatchSize, self.state_size))
        Nextstates = np.zeros((self.BatchSize, self.state_size))
        reward, action, done = [], [], []


        for i in range(self.BatchSize):
            Nowstates[i] = mini_batch_set[i][0]
            action.append(mini_batch_set[i][1])
            reward.append(mini_batch_set[i][2])
            Nextstates[i] = mini_batch_set[i][3]
            done.append(mini_batch_set[i][4])

        target = self.Behavior_model.predict(Nowstates)
        target_val = self.Target_model.predict(Nextstates)

        for i in range(self.BatchSize):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.discount_rate * (np.amax(target_val[i]))

        self.Behavior_model.fit(Nowstates, target,batch_size=self.BatchSize, epochs=1, verbose=0)

    def Target_model_Update(self):
        self.Target_model.set_weights(self.Behavior_model.get_weights())

    def load_model(self,path):
        self.Behavior_model.load_weights(path)


if __name__ == "__main__":
    env = gym.make("MountainCar-v0")
    env.reset()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size, False)

    scores = []
    Episode = []

    for episode in range(Num_Episode):
        done = False
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        score = 0

        while not done:
            env.render()

            action = agent.DoPolicyAction(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            if next_state[0][0] >= 0.5:
                reward = 5
            agent.Save_replay_memory(state, action, reward, next_state, done)

            if (len(agent.Replay_memory)) > agent.Train_start_Replay_memery_size :
                agent.Train_Model()
            
            score += reward
            state = next_state

            if done:

                agent.Target_model_Update()
                scores.append(score)
                Episode.append(episode)
                pylab.plot(Episode, scores, 'b')
                pylab.savefig("D:\\RL_Study\\Model_save\\save_grpah\\MountainCar_dqn.png")
                print("episode:", episode, "  score:", score, "  memory length:", len(agent.Replay_memory), "  epsilon:", agent.epsilon)

                if np.mean(scores[-min(10, len(scores)):]) > 490:
                    agent.Behavior_model.save_weights("D:\\RL_Study\\Model_save\\cartpole_dqn.h5")
                    sys.exit()



