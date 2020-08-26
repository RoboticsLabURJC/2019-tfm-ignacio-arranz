import time
import datetime
import pickle

import gym
import liveplot
import gym_gazebo
import numpy as np
from gym import logger, wrappers
from qlearn import QLearn

import agents.f1.settings as settings
from agents.f1.settings import actions_set


def render():
    render_skip = 0  # Skip first X episodes.
    render_interval = 50  # Show render Every Y episodes.
    render_episodes = 10  # Show Z episodes every rendering.

    if (episode % render_interval == 0) and (episode != 0) and (episode > render_skip):
        env.render()
    elif ((episode - render_episodes) % render_interval == 0) and (episode != 0) and (episode > render_skip) and \
            (render_episodes < episode):
        env.render(close=True)


def load_model(qlearn, file_name):

    qlearn_file = open("logs/qlearn_models/" + file_name)
    model = pickle.load(qlearn_file)

    qlearn.q = model
    qlearn.alpha = settings.algorithm_params["alpha"]
    qlearn.gamma = settings.algorithm_params["gamma"]
    qlearn.epsilon = settings.algorithm_params["epsilon"]
    # highest_reward = settings.algorithm_params["highest_reward"]

    print("\n\nMODEL LOADED. Number of (action, state): {}\n\n".format(len(model)))
    print("    - Len: {}".format(len(qlearn.q)))


def save_model(environmen, epoch):
    # Tabular RL: Tabular Q-learning basically stores the policy (Q-values) of  the agent into a matrix of shape
    # (S x A), where s are all states, a are all the possible actions. After the environment is solved, just save this
    # matrix as a csv file. I have a quick implementation of this on my GitHub under Reinforcement Learning.
    date = datetime.datetime.now()
    format = date.strftime("%Y%m%d_%H%M")
    file_name = "_qlearn_circuit_{}_act_set_{}_e_{}_epoch_{}".format(environment["circuit_name"],
                                                                     settings.actions_set,
                                                                     round(qlearn.epsilon, 2),
                                                                     epoch)
    file = open("logs/qlearn_models/" + format + file_name + '.pkl', 'wb')
    pickle.dump(qlearn.q, file)


####################################################################################################################
# MAIN PROGRAM
####################################################################################################################
if __name__ == '__main__':

    print(settings.title)
    print(settings.description)

    environment = settings.envs_params["simple"]
    env = gym.make(environment["env"])

    outdir = './logs/f1_qlearn_gym_experiments/'
    stats = {}  # epoch: steps

    env = gym.wrappers.Monitor(env, outdir, force=True)
    plotter = liveplot.LivePlot(outdir)

    last_time_steps = np.ndarray(0)

    actions = range(env.action_space.n)

    stimate_step_per_lap = 4000
    lap_completed = False

    qlearn = QLearn(actions=actions, alpha=0.2, gamma=0.9, epsilon=0.99)

    if settings.load_model:
        solution = "qlearn_camera_resuelto/20200814_191734_qlearn_model_e_0.9801_a_0.2_g_0.9.pkl"
        file_name = '20200824_204057_qlearn_circuit_nurburgring_act_set_medium_e_0.13_epoch_2000.pkl'
        load_model(qlearn, solution)

        highest_reward = max(qlearn.q.values(), key=stats.get)
    else:
        highest_reward = 0
        initial_epsilon = qlearn.epsilon

    total_episodes = 20000
    epsilon_discount = 0.999  # Default 0.9986

    start_time = time.time()

    print(settings.lets_go)

    for episode in range(total_episodes):

        done = False
        lap_completed = False

        cumulated_reward = 0  # Should going forward give more reward then L/R z?

        observation = env.reset()

        if qlearn.epsilon > 0.05:
            qlearn.epsilon *= epsilon_discount

        # render()  # defined above, not env.render()

        state = ''.join(map(str, observation))

        # print("-------- RESET: {}".format(state))
        # print("DICCIONARIO ----> {}".format(len(qlearn.q)))

        for step in range(20000):

            # Pick an action based on the current state
            action = qlearn.selectAction(state)

            # Execute the action and get feedback
            observation, reward, done, info = env.step(action)
            cumulated_reward += reward

            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            nextState = ''.join(map(str, observation))
            # print("-------- {}".format(nextState))

            qlearn.learn(state, action, reward, nextState)

            env._flush(force=True)

            if not done:
                state = nextState
            else:
                last_time_steps = np.append(last_time_steps, [int(step + 1)])
                stats[episode] = step
                break

            if stimate_step_per_lap > 4000 and not lap_completed:
                print("LAP COMPLETED!!")
                lap_completed = True

            # print("Obser: {} - Rew: {}".format(observation, reward))

        if episode % 100 == 0 and settings.plotter_graphic:
            # plotter.plot(env)
            plotter.plot_steps_vs_epoch(stats)
            # plotter.full_plot(env, stats, 2)  # optional parameter = mode (0, 1, 2)

        if episode % 500 == 0 and settings.save_model and episode > 1:
            print("\nSaving model . . .\n")
            save_model(environment, episode)

        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        print ("EP: " + str(episode + 1) + " - epsilon: " + str(round(qlearn.epsilon, 2)) + " - Reward: " + str(
            cumulated_reward) + " - Time: %d:%02d:%02d" % (h, m, s) + " - steps: " + str(step))

    print ("\n|" + str(total_episodes) + "|" + str(qlearn.alpha) + "|" + str(qlearn.gamma) + "|" + str(
        initial_epsilon) + "*" + str(epsilon_discount) + "|" + str(highest_reward) + "| PICTURE |")

    l = last_time_steps.tolist()
    l.sort()

    # print("Parameters: a="+str)
    print("Overall score: {:0.2f}".format(last_time_steps.mean()))
    print("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

    env.close()
