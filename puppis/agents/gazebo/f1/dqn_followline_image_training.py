import os
import gym
import time
import gym_pyxis
import collections
import random
import numpy as np

import tensorflow as tf

from utils import wrappers
from utils import dqn_model


MEAN_REWARD_BOUND = 800

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 10000

EPSILON_DECAY_LAST_FRAME = 10**5
EPSILON_START = 1.0
EPSILON_FINAL = 0.02


Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(next_states)


class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = env.reset()
        self.total_reward = 0.0

    def play_step(self, net, epsilon=0.0):
        done_reward = None

        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            state_v = tf.convert_to_tensor(state_a)
            q_vals_v = net(state_v)
            act_v = tf.argmax(q_vals_v, axis=1, output_type=tf.int32)
            action = int(act_v.numpy())

        # do step in the environment
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward
        new_state = new_state

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state

        if is_done:
            done_reward = self.total_reward
            self._reset()

        return done_reward


def calc_loss(batch, net, tgt_net):
    states, actions, rewards, dones, next_states = batch

    states_v = tf.convert_to_tensor(states)
    next_states_v = tf.convert_to_tensor(next_states)
    actions_v = tf.convert_to_tensor(actions)
    rewards_v = tf.convert_to_tensor(rewards)

    state_action_values = tf.squeeze(tf.batch_gather(net(states_v), tf.expand_dims(actions_v, -1)), -1)
    next_state_values = tf.reduce_max(tgt_net(next_states_v), axis=1)

    next_state_values = tf.where(dones == 0, next_state_values, tf.zeros_like(next_state_values))
    expected_state_action_values = next_state_values * GAMMA + rewards_v

    return tf.losses.mean_squared_error(state_action_values, expected_state_action_values)


if __name__ == "__main__":

    writer = tf.contrib.summary.create_file_writer(logdir='runs', flush_millis=10000, filename_suffix="-dqn-f1-followline")

    env = wrappers.make_env('F1FollowLineCameraEnv-v0')
    env.load_checkpoints('adjusted_checkpoints.json')
    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START

    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_mean_reward = None

    state = env.reset()

    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n)
    tgt_net = dqn_model.DQN(env.observation_space.shape, env.action_space.n)

    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    global_step = tf.Variable(0)

    checkpoint_dir = 'checkpoints/'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    root = tf.train.Checkpoint(optimizer=optimizer,
                               model=net,
                               optimizer_step=tf.train.get_or_create_global_step())

    while True:
        frame_idx += 1
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)

        reward = agent.play_step(net, epsilon)
        if reward is not None:
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            mean_reward = np.mean(total_rewards[-100:])
            print("%d: done %d games, mean reward %.3f, eps %.2f, speed %.2f f/s" % (
                frame_idx, len(total_rewards), mean_reward, epsilon,
                speed
            ))
            with writer.as_default(), tf.contrib.summary.always_record_summaries():
                tf.contrib.summary.scalar("epsilon", epsilon)
                tf.contrib.summary.scalar("speed", speed)
                tf.contrib.summary.scalar("reward_100", mean_reward)
                tf.contrib.summary.scalar("reward", reward)

            if best_mean_reward is None or best_mean_reward < mean_reward:
                root.save(checkpoint_prefix)
                if best_mean_reward is not None:
                    print("Best mean reward updated %.3f -> %.3f, model saved" % (best_mean_reward, mean_reward))
                best_mean_reward = mean_reward
            if mean_reward > MEAN_REWARD_BOUND:
                print("Solved in %d frames!" % frame_idx)
                break

        if len(buffer) < REPLAY_START_SIZE:
            continue

        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.set_weights(net.get_weights())

        batch = buffer.sample(BATCH_SIZE)
        with tf.GradientTape() as tape:
            loss_value = calc_loss(batch, net, tgt_net)

        grads = tape.gradient(loss_value, net.trainable_variables)

        optimizer.apply_gradients(zip(grads, net.trainable_variables), global_step)

    writer.close()