import gym
import numpy as np


def compute_value_function(gym_env, policy, gamma=1.0):
    value_table = np.zeros(gym_env.env.nS)
    threshold = 1e-10

    while True:

        updated_value_table = np.copy(value_table)
        for state in range(env.env.nS):
            action = policy[state]

            value_table[state] = sum([trans_prob * (reward_prob + gamma * updated_value_table[next_state])
                                      for trans_prob, next_state, reward_prob, _ in gym_env.env.P[state][action]])

        if np.sum((np.fabs(updated_value_table - value_table))) <= threshold:
            break

    return value_table


def extract_policy(gym_env, value_table, gamma=1.0):
    policy = np.zeros(gym_env.observation_space.n)
    for state in range(gym_env.observation_space.n):
        Q_table = np.zeros(gym_env.action_space.n)
        for action in range(gym_env.action_space.n):
            for next_sr in gym_env.env.P[state][action]:
                trans_prob, next_state, reward_prob, _ = next_sr
                Q_table[action] += (trans_prob * (reward_prob + gamma * value_table[next_state]))

        policy[state] = np.argmax(Q_table)

    return policy


def policy_iteration(gym_env, gamma=1.0):
    random_policy = np.zeros(gym_env.observation_space.n)
    iterations = 200000
    for i in range(iterations):
        new_value_function = compute_value_function(gym_env, random_policy, gamma)
        new_policy = extract_policy(gym_env, new_value_function, gamma)

        if np.all(random_policy == new_policy):
            print('Policy iteration converged at step {}.'.format(i+1))
            break
        random_policy = new_policy
    return new_policy


if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')
    result = policy_iteration(env, 0.8)
    print(result)

