from os import stat
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from Actor import Actor
from Critic import Critic

class agent:
    def __init__(self, state_space, action_space, N_batch, discount, LR, step=1):
        self.actor = Actor(state_space=state_space, action_space=action_space)
        self.critic = Critic(state_space=state_space)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
        self.replay_batch = []
        self.num_batch = N_batch
        self.step = step
        self.discount = discount

    def get_action(self, state):
        state = np.expand_dims(state, axis=0)
        action_prob = self.actor(state)
        action_prob = action_prob.numpy()
        dist = tfp.distributions.Categorical(probs=action_prob, dtype=tf.float32)
        action = dist.sample()
        return int(action.numpy()[0])

    def update(self, done):
        for t in range(len(self.replay_batch)):
            s, a, r, n_s = self.replay_batch[t]
            state = np.array([s])
            next_state = np.array([n_s])
            with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
                prob = self.actor(state, training=True)
                value = self.critic(state, training=True)
                value_next = self.critic(next_state, training=True)
                advantage = r +  (1 - (1 if done else 0)) * (self.discount * value_next - value)
                dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
                actor_loss = -1 * dist.log_prob(a) * advantage
                critic_loss = advantage ** 2
            grad_actor = tape1.gradient(actor_loss, self.actor.trainable_variables)
            grad_critic = tape2.gradient(critic_loss, self.critic.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(grad_actor, self.actor.trainable_variables))
            self.critic_optimizer.apply_gradients(zip(grad_critic, self.critic.trainable_variables))
        self.replay_batch = []
        return actor_loss, critic_loss

    def save_batch(self, state, action, reward, n_state, done):
        self.replay_batch.append((state, action, reward, n_state))
        if len(self.replay_batch) >= self.num_batch:
            self.update(done=done)