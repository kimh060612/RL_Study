import numpy as np
import tensorflow as tf
from tensorflow import keras

class Actor(keras.Model): # Policy Estimation
    def __init__(self, state_space, action_space):
        super().__init__()
        self.layer1 = keras.layers.Dense(64, activation='relu', kernel_initializer='random_normal', input_dim=state_space)
        self.layer2 = keras.layers.Dropout(0.2)
        self.layer3 = keras.layers.Dense(32, kernel_initializer='random_normal', activation='relu')
        self.layer4 = keras.layers.Dropout(0.2)
        self.layer5 = keras.layers.Dense(16, kernel_initializer='random_normal', activation='relu')
        self.layer6 = keras.layers.Dense(action_space, kernel_initializer='random_normal', activation='softmax')
    
    def call(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x
            

            