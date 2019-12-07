import gym
import numpy as np


"""
        S F F F
        F H F H 
        F F F H 
        H F F G 
        
    S : starting point, safe
    F : frozen surface, safe
    H : hole, fall to your doom
    G : goal, where the frisbee is located
"""

def value_itearation(gym_env, gamma=1.0):
    value_table = np.zeros(gym_env.observation_space.n)
    n_iterations = 10000
    threshold = 1e-20

    for i in range(n_iterations):
        updated_value_table = np.copy(value_table)
        for state in range(gym_env.observation_space.n):
            Q_value = []
            for action in range(gym_env.action_space.n):
                next_states_rewards = []
                for next_sr in gym_env.env.P[state][action]:
                    trans_prob, next_state, reward_prob, _ = next_sr
                    next_states_rewards.append((trans_prob * (reward_prob + gamma * updated_value_table[next_state])))

                Q_value.append(np.sum(next_states_rewards))
            value_table[state] = max(Q_value)

        if np.sum(np.fabs(updated_value_table - value_table)) <= threshold:
            print('Value-iteration converged at iteration# {}'.format((i+1)))

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


if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')
    optimal_value_function = value_itearation(env, 0.0)
    optimal_policy = extract_policy(env, optimal_value_function, 0.0)
    print(optimal_policy)
