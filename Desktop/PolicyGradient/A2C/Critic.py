import numpy as np
import tensorflow as tf
from tensorflow import keras

class Critic(keras.Model): # Value Function Estimation
    def __init__(self, state_space):
        super().__init__()

        self.layer1 = keras.layers.Dense(128, activation='relu', kernel_initializer='random_normal', input_dim=state_space)
        #self.layer2 = keras.layers.Dropout(0.2)
        self.layer3 = keras.layers.Dense(64, kernel_initializer='random_normal', activation='relu')
        self.layer4 = keras.layers.Dense(1, kernel_initializer='random_normal', activation=None)

    def call(self, state):
        x = self.layer1(state)
        # x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x