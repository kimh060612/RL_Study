import numpy as np
import tensorflow as tf
from tensorflow import keras

LR = pow(2, -16)
optimizer = tf.keras.optimizers.Adam(learning_rate=LR)

class Critic: # Value Function Estimation
    def __init__(self, state_space, discount = 0.99):
        self.discount = discount
        self.layer1 = keras.layers.Dense(128, activation='relu', kernel_initializer='random_normal', input_dim=state_space)
        self.layer2 = keras.layers.Dropout(0.2)
        self.layer3 = keras.layers.Dense(64, kernel_initializer='random_normal', activation='relu')
        self.layer4 = keras.layers.Dense(1, kernel_initializer='random_normal', activation='relu')

    def forward(self, state):
        x = self.layer1(state)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def get_advantage(self, state, reward, n_state):
        return reward + self.discount * self.forward(np.expand_dims(n_state, axis=0)) - self.forward(np.expand_dims(state, axis=0))

    def update(self):
        with tf.GradientTape() as tape:
            
            pass