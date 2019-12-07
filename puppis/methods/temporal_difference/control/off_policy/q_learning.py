import numpy as np
from collections import defaultdict


class QLearning(object):

    def __init__(self, actions, epsilon=0.01, alpha=0.5, gamma=1):
        '''
        Constructor for QLearning class

        :param actions: list of actions allowed
        :param epsilon: threshold value for epsilon-greedy policy
        :param alpha: learning rate
        :param gamma: discount factor
        '''
        self.actions = actions
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.Q = defaultdict(lambda : [0.0] * len(actions))

    def learn(self, state, action, reward, next_state):
        '''
        <S, A, R, S'>
        Q-learning Update:
        Q(S,A) = Q(S,A) + alpha[R + gamma max(Q(S',A)) - Q(S,A)]

        :param state: actual state
        :param action: action selected
        :param reward: value of reward
        :param next_state: next state
        '''

        current_q_value = self.Q[state][action]
        max_next_action_q_value = max(self.Q[next_state])
        self.Q[state][action] = current_q_value + (self.alpha * (reward + (self.gamma * (max_next_action_q_value - current_q_value))))


    def get_action(self, state):
        '''
        Given a state, return an action using epsilon-greedy policy

        :param state: state
        :return: action
        '''

        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.actions)
        else:
            max_value_action = np.amax(self.Q[state])
            max_value_action_index = np.where(self.Q[state] == max_value_action)
            action = np.random.choice(max_value_action_index[0])

        return action

