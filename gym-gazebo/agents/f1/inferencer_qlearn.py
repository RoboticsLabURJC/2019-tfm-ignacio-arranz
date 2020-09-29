import os
import time
import datetime
import pickle

import gym
import gym_gazebo
import numpy as np
from gym import wrappers
from qlearn import QLearn

import agents.f1.settings as settings


def load_model(actions, input_dir, experiment, number, filename):

    qlearn_file = open(os.path.join(input_dir, experiment, number, filename))
    model = pickle.load(qlearn_file)

    qlearn = QLearn(actions=actions, alpha=0.8, gamma=0.9, epsilon=0.05)
    qlearn.q = model

    print("-----------------------\nMODEL LOADED\n-----------------------")
    return qlearn

####################################################################################################################
# MAIN PROGRAM
####################################################################################################################
if __name__ == '__main__':

    print(settings.title)
    print(settings.description)
    print("    - Start hour: {}".format(datetime.datetime.now()))

    environment = settings.envs_params["simple"]
    env = gym.make(environment["env"])

    input_dir = './logs/qlearn_models/qlearn_camera_solved'
    experiment = 'points_1_actions_simple__simple_circuit'
    number = '4'
    filename = "1_20200921_2024_act_set_simple_epsilon_0.83_QTABLE.pkl"

    actions = range(env.action_space.n)

    last_time_steps = np.ndarray(0)
    counter = 0
    highest_reward = 0
    epsilon_discount = 0.98  # Default 0.9986
    stimate_step_per_lap = 4000
    lap_completed = False
    total_episodes = 5

    qlearn = load_model(actions, input_dir, experiment, number, filename)

    telemetry_start_time = time.time()
    start_time = datetime.datetime.now()
    start_time_format = start_time.strftime("%Y%m%d_%H%M")

    print(settings.lets_go)

    for episode in range(total_episodes):
        counter = 0
        done = False
        lap_completed = False

        cumulated_reward = 0  # Should going forward give more reward then L/R z?

        observation = env.reset()
        state = ''.join(map(str, observation))

        for step in range(500000):

            counter += 1

            # Pick an action based on the current state
            action = qlearn.selectAction(state)

            # Execute the action and get feedback
            observation, reward, done, info = env.step(action)
            cumulated_reward += reward

            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            nextState = ''.join(map(str, observation))
            # qlearn.learn(state, action, reward, nextState)
            # env._flush(force=True)

            if not done:
                state = nextState
            else:
                print("\n\n\n ---> Try: {}/{}".format(episode+1, total_episodes))
                last_time_steps = np.append(last_time_steps, [int(step + 1)])
                break

            if counter > 1000:
                qlearn.epsilon *= epsilon_discount
                time = datetime.datetime.now() - start_time
                print("Checkpoint")
                print("\t- cum reward: {}\n\t- time: {}\n\t- steps: {}\n".format(cumulated_reward, time, step))
                counter = 0

            if datetime.datetime.now() - datetime.timedelta(minutes=20) > start_time:
                print(settings.race_completed)
                print("    - N epoch:     {}".format(episode))
                print("    - Model size:  {}".format(len(qlearn.q)))
                print("    - Action set:  {}".format(settings.actions_set))
                print("    - Epsilon:     {}".format(round(qlearn.epsilon, 2)))
                print("    - Cum. reward: {}".format(cumulated_reward))

                env.close()
                exit(0)

    print("TOO MANY ATTEMPTS. NO SUCCESS")

    env.close()
    exit(0)
