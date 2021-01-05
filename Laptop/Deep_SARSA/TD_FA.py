import gym
import keras.backend as K
import numpy as np
from keras.optimizers import Adam
from keras.layers import Dense

class TDagent:
    def __init__(self, actionNum, stateNum):
                
        pass

if __name__ == "__main__":
    ENV = gym.make("MountainCar-v0")
    num_action = ENV.action_space.n
    num_state = ENV.observation_space.shape[0]



    pass