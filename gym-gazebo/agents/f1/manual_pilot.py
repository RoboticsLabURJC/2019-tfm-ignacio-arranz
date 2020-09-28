#!/usr/bin/env python
# coding: utf-8

import time
import datetime

import gym
import gym_gazebo
from gym import wrappers


import agents.f1.settings as settings
from gym_gazebo.envs.f1.env_manual_pilot import title

total_episodes = 20000




if __name__ == '__main__':

    print(title)

    print("    - Start hour: {}".format(datetime.datetime.now()))

    environment = settings.envs_params["manual"]
    env = gym.make(environment["env"])

    previous = datetime.datetime.now()


    time.sleep(5)
    for episode in range(total_episodes):

        now = datetime.datetime.now()
        if now - datetime.timedelta(seconds=3) > previous:
            previous = datetime.datetime.now()
            env.store_position()

        env.execute()




