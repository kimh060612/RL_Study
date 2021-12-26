from os import stat
import numpy as np
import tensorflow as tf
from .Actor import Actor
from .Critic import Critic

class agent:
    def __init__(self, state_space, action_space, N_batch):
        self.actor = Actor(state_space=state_space, action_space=action_space)
        self.critic = Critic(state_space=state_space)
        self.replay_batch = []
        self.num_batch = N_batch

    def get_action(self, state):
        return self.actor.get_action(state)

    def update(self):
        for t in range(len(self.replay_batch)):
            s, a, r, n_s = self.replay_batch[t]
            
        self.replay_batch = []

    def save_batch(self, state, action, reward, n_state):
        self.replay_batch.append((state, action, reward, n_state))
        if len(self.replay_batch) >= self.num_batch:
            self.update()
            self.replay_batch = []