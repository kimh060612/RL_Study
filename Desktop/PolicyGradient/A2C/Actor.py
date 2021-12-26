import numpy as np
import tensorflow as tf
from tensorflow import keras

LR = pow(2, -16)
optimizer = tf.keras.optimizers.Adam(learning_rate=LR)

class Actor: # Policy Estimation
    def __init__(self, state_space, action_space):
        
        self.layer1 = keras.layers.Dense(64, activation='relu', kernel_initializer='random_normal', input_dim=state_space)
        self.layer2 = keras.layers.Dropout(0.2)
        self.layer3 = keras.layers.Dense(32, kernel_initializer='random_normal', activation='relu')
        self.layer4 = keras.layers.Dropout(0.2)
        self.layer5 = keras.layers.Dense(16, kernel_initializer='random_normal', activation='relu')
        self.layer6 = keras.layers.Dense(action_space, kernel_initializer='random_normal', activation='softmax')
    
    def forward(self, state):
        x = self.layer1(state)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x
    
    def get_action(self, state):
        action_prob = self.forward(np.expand_dims(state, axis=0))
        action = tf.argmax(action_prob[0])
        return action.numpy()

    def update(self):
        with tf.GradientTape() as tape:
            loss = 0
            

            