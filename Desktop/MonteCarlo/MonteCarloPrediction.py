import gym
import numpy as np
import random

GAMMA=0.9
EPISODES=1000
POLICY = [0.25, 0.25, 0.25, 0.25]
THRESHOLD = 1e-20

env = gym.make('FrozenLake-v0')

if __name__ == "__main__":
    
    pass