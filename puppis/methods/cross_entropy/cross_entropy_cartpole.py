from __future__ import absolute_import, division, print_function
import gym
from gym import wrappers
import numpy as np
from collections import namedtuple

import tensorflow as tf

tf.enable_eager_execution()

Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])


class Net(tf.keras.Model):

    def __init__(self, hidden_size, obs_size, n_actions):
        super(Net, self).__init__()
        self.dense1 = tf.keras.layers.Dense(hidden_size, activation=tf.nn.relu, input_shape=(obs_size,))
        self.dense2 = tf.keras.layers.Dense(n_actions)

    def call(self, input):
        result = self.dense1(input)
        result = self.dense2(result)
        return result


def iterate_batches(env, net, batch_size):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs = env.reset()

    while True:
        obs_v = tf.convert_to_tensor([obs])
        net_output = net(obs_v)
        act_probs_v = tf.nn.softmax(net_output)
        act_probs = act_probs_v.numpy()[0]
        action = np.random.choice(len(act_probs), p=act_probs)
        next_obs, reward, is_done, _ = env.step(action)
        episode_reward += reward
        episode_steps.append(EpisodeStep(observation=obs, action=action))
        if is_done:
            batch.append(Episode(reward=episode_reward, steps=episode_steps))
            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset()
            if len(batch) == batch_size:
                yield batch
                batch = []
        obs = next_obs


def play(net, env, iterations=1000):
    episode_reward = 0.0
    obs = env.reset()

    for _ in range(iterations):
        env.render()
        obs_v = tf.convert_to_tensor([obs])
        net_output = net(obs_v)
        act_probs_v = tf.nn.softmax(net_output)
        act_probs = act_probs_v.numpy()[0]
        action = tf.argmax(act_probs, output_type=tf.int32)
        next_obs, reward, is_done, _ = env.step(int(action.numpy()))
        episode_reward += reward
        if is_done:
            print("The episode concluded, reward: {}".format(episode_reward))
            episode_reward = 0
            next_obs = env.reset()
        obs = next_obs

    env.close()


def filter_batch(batch, percentile):
    rewards = list(map(lambda s: s.reward, batch))
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))

    train_obs = []
    train_act = []
    for example in batch:
        if example.reward < reward_bound:
            continue
        train_obs.extend(map(lambda step: step.observation, example.steps))
        train_act.extend(map(lambda step: step.action, example.steps))

    train_obs_v = tf.convert_to_tensor(train_obs)
    train_act_v = tf.convert_to_tensor(train_act)
    return train_obs_v, train_act_v, reward_bound, reward_mean


def loss(model, x, y):
    y_ = model(x)
    return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


if __name__ == "__main__":
    hidden_size = 128
    batch_size = 16
    percentile = 70

    env = gym.make("CartPole-v0")

    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    writer = tf.contrib.summary.create_file_writer(logdir='runs', flush_millis=10000, filename_suffix="-cartpole")
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    global_step = tf.Variable(0)
    net = Net(hidden_size, obs_size, n_actions)

    for iter_no, batch in enumerate(iterate_batches(env, net, batch_size)):
        obs_v, acts_v, reward_b, reward_m = filter_batch(batch, percentile)
        loss_v, grads = grad(net, obs_v, acts_v)
        optimizer.apply_gradients(zip(grads, net.trainable_variables), global_step)

        print("%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f" % (
            iter_no, loss_v.numpy(), reward_m, reward_b))

        with writer.as_default(), tf.contrib.summary.always_record_summaries():
            tf.contrib.summary.scalar("loss", loss_v.numpy())
            tf.contrib.summary.scalar("reward_bound", reward_b)
            tf.contrib.summary.scalar("reward_mean", reward_m)

        if reward_m > 199:
            print("Solved!")
            break

    env.close()
    writer.close()

    env = gym.make('CartPole-v0')
    env = wrappers.Monitor(env, '/tmp/cartpole-cross-entropy', force=True)
    play(net, env)
    env.close()