#!/usr/bin/env python3
import gym
import time
import argparse
import numpy as np

import tensorflow as tf

from utils import dqn_model
from utils import wrappers

import collections

LEARNING_RATE = 1e-4
DEFAULT_ENV_NAME = "F1FollowLineCameraEnv-v0"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Model file to load")
    parser.add_argument("-e", "--env", default=DEFAULT_ENV_NAME,
                        help="Environment name to use, default=" + DEFAULT_ENV_NAME)
    parser.add_argument("-r", "--record", help="Directory to store video recording")
    parser.add_argument("--no-visualize", default=True, action='store_false', dest='visualize',
                        help="Disable visualization of the game play")
    args = parser.parse_args()

    env = wrappers.make_env(args.env)
    if args.record:
        env = gym.wrappers.Monitor(env, args.record)
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n)

    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    root = tf.train.Checkpoint(optimizer=optimizer,
                               model=net,
                               optimizer_step=tf.train.get_or_create_global_step())

    root.restore(tf.train.latest_checkpoint(args.model))

    state = env.reset()
    total_reward = 0.0
    c = collections.Counter()

    while True:
        start_ts = time.time()

        state_v = tf.convert_to_tensor(np.array([state], copy=False))
        q_vals_v = net(state_v)
        act_v = tf.argmax(q_vals_v, axis=1, output_type=tf.int32)
        action = int(act_v.numpy())

        c[action] += 1
        state, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break

    print("Total reward: %.2f" % total_reward)
    print("Action counts:", c)
    if args.record:
        env.env.close()

