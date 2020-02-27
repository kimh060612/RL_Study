import numpy as np
import random
from environment import Env
from collections import defaultdict

EpisodeNum = 5000

class QAgent:
    
    def __init__(self, actions):
        self.actions = actions
        self.learning_rate = 0.01
        self.discount_factor = 0.9
        self.epsilon = 0.9
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    
    def DoAction(self, state):
        action = 0 # initialize (No Meaning)
        # Epsilon Greedy Policy
        if (np.random.rand() < self.epsilon): 
            action = np.random.choice(self.actions)
        else :
            state_action_list = self.q_table[state]
            action = self.arg_max(state_action_list)
        return action
    
    def Update(self, Nowstate, action, reward,Nextstate):
        Q_now = self.q_table[Nowstate][action]
        Q_target = reward + self.discount_factor * max(self.q_table[Nextstate])
        self.q_table[Nowstate][action] += self.learning_rate * (Q_target - Q_now)

    def arg_max(self, state_action):
        max_index_list = []
        max_value = state_action[0]
        for index, value in enumerate(state_action):
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value:
                max_index_list.append(index)
        return random.choice(max_index_list)

if __name__ == "__main__":
    env = Env()
    action_list = list(range(env.n_actions))
    agent = QAgent(action_list)

    for epis in range(EpisodeNum):
        state = env.reset()

        while True:
            env.render()

            action = agent.DoAction(str(state))
            next_state, reward, IS_done = env.step(action)
            
            agent.Update(str(state), action, reward, str(next_state))
            state = next_state

            env.print_value_all(agent.q_table)

            if IS_done:
                break
