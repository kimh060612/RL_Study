import tensorflow as tf
import gym
import keras.backend as K
from keras.optimizers import Adam
from keras.layers import Dense
from keras.models import Sequential

Num_Episode = 5000

class DQNAgnet:
    def __init__(self, StateNum, ActionNum):
        self.size_state = StateNum
        self.size_action = ActionNum
        self.learning_rate = 0.001

    
    def Evaluation(self):
        
        pass

    def Policy_Improvement(self):

        pass

    def NeuralNet_Model(self): #Neural Network Function Approximation
        model = Sequential()
        model.add(Dense(24, input_dim=self.size_state, activation='relu',kernel_initializer='he_uniform'))
        model.add(Dense(24, activation='relu',kernel_initializer='he_uniform'))
        model.add(Dense(self.size_action, activation="linear",kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def Memory_save(self):
        pass

    def Memory_replay(self):
        pass

    def load_model(self):
        pass


if __name__ == "__main__":

    pass