import gym
import random

class Environment:
    def __init__(self):
        self.steps_left = 10

    def getObservation(self):
        return [ 0.0, 0.0, 0.0 ]
    
    def isDone(self) :
        return self.steps_left == 0

    def action(self, action):
        if self.isDone():
            raise Exception("Game is Over")
        self.steps_left -= 1
        return random.random()

class Agent:
    def __init__(self):
        self.total_reward = 0.0
    
    def step(self):
        
        pass