from collections import deque
import gym
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from puppis.methods.temporal_difference.control.on_policy.sarsa import SARSA


if __name__ == "__main__":
    sns.set()
    scores = deque(maxlen=100)
    n_episodes = 1000
    env = gym.make('CartPole-v0')

    actions = range(env.action_space.n)

    sarsa = SARSA(actions)

    cumulative_reward = np.array(range(n_episodes))
    steps_per_episode = np.array(range(n_episodes))

    for e in range(n_episodes):
        current_state = str(env.reset())
        action = sarsa.get_action(current_state)
        done = False
        G = 0.0
        steps = 0

        while not done:
            env.render(mode='human')

            next_obs, reward, done, info = env.step(action)
            next_state = str(next_obs)
            next_action = sarsa.get_action(next_state)

            sarsa.learn(current_state, action, reward, next_state, next_action)

            current_state = next_state
            action = next_action
            steps += 1
            G += reward

        cumulative_reward[e] = G
        steps_per_episode[e] = steps
        scores.append(steps)

        mean_score = np.mean(scores)
        if e % 100 == 0:
            print('[Episode {}] - Mean survival time over last 100 episodes was {} ticks.'.format(e, mean_score))

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.plot(np.array(range(n_episodes)), steps_per_episode)
    ax1.set_ylabel('episode length')

    ax2 = fig.add_subplot(212)
    ax2.plot(np.array(range(n_episodes)), cumulative_reward)
    ax2.set_xlabel('episodes')
    ax2.set_ylabel('episode reward')

    plt.show()

    env.close()
