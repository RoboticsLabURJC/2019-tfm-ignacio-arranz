import gym
import rospy
import roslaunch
import time
import numpy as np

from gym import utils, spaces
from gym.utils import seeding
from gym_gazebo.envs import gazebo_env
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState, SetModelState

from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from sensor_msgs.msg import LaserScan


class GazeboF1QlearnLaserEnv(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "F1Lasercircuit_v0.launch")
        self.vel_pub = rospy.Publisher('/F1ROS/cmd_vel', Twist, queue_size=5)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.action_space = spaces.Discrete(3)  # F, L, R
        self.reward_range = (-np.inf, np.inf)
        self._seed()

    def render(self, mode='human'):
        pass

    @staticmethod
    def discrete_observation(data, new_ranges):

        discrete_ranges = []
        min_range = 0.05
        done = False
        mod = len(data.ranges) / new_ranges
        new_data = data.ranges[10:-10]
        for i, item in enumerate(new_data):
            if i % mod == 0:
                if data.ranges[i] == float('Inf') or np.isinf(data.ranges[i]):
                    discrete_ranges.append(6)
                elif np.isnan(data.ranges[i]):
                    discrete_ranges.append(0)
                else:
                    discrete_ranges.append(int(data.ranges[i]))
            if min_range > data.ranges[i] > 0:
                # print("Data ranges: {}".format(data.ranges[i]))
                done = True
                break

        return discrete_ranges, done

    @staticmethod
    def calculate_observation(data):
        min_range = 0.5  # Default: 0.21
        done = False
        for i, item in enumerate(data.ranges):
            if min_range > data.ranges[i] > 0:
                done = True
        return done

    @staticmethod
    def get_center_of_laser(data):

        laser_len = len(data.ranges)
        left_sum = sum(data.ranges[laser_len - (laser_len / 5):laser_len - (laser_len / 10)])  # 80-90
        right_sum = sum(data.ranges[(laser_len / 10):(laser_len / 5)])  # 10-20

        center_detour = (right_sum - left_sum) / 5

        return center_detour

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except rospy.ServiceException as e:
            print(e)
            print("/gazebo/unpause_physics service call failed")

        if action == 0:  # FORWARD
            vel_cmd = Twist()
            vel_cmd.linear.x = 3
            vel_cmd.angular.z = 0.0
            self.vel_pub.publish(vel_cmd)
        elif action == 1:  # LEFT
            vel_cmd = Twist()
            vel_cmd.linear.x = 3  # 0.05
            vel_cmd.angular.z = 1  # 0.3
            self.vel_pub.publish(vel_cmd)
        elif action == 2:  # RIGHT
            vel_cmd = Twist()
            vel_cmd.linear.x = 3  # 0.05
            vel_cmd.angular.z = -1  # -0.3
            self.vel_pub.publish(vel_cmd)

        laser_data = None
        while laser_data is None:
            laser_data = rospy.wait_for_message('/F1ROS/laser/scan', LaserScan, timeout=5)

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except rospy.ServiceException as e:
            print(e)
            print("/gazebo/pause_physics service call failed")

        state, _ = self.discrete_observation(laser_data, 5)

        laser_len = len(laser_data.ranges)
        left_sum = sum(laser_data.ranges[laser_len - (laser_len / 5):laser_len - (laser_len / 10)])  # 80-90
        right_sum = sum(laser_data.ranges[(laser_len / 10):(laser_len / 5)])  # 10-20

        center_detour = (right_sum - left_sum) / 5
        done = False
        if abs(center_detour) > 2:
            done = True
        # print("center: {}".format(center_detour))

        # 3 actions
        # if not done:
        #     if abs(center_detour) < 2:
        #          reward = 1 / float(center_detour + 1)
        #     else:  # L or R no looping
        #          reward = 0.5 / float(center_detour + 1)
        # else:
        #     reward = -200

        if not done:
            if action == 0:
                reward = 5
            else:
                reward = 1
        else:
            reward = -200

        return state, reward, done, {}

    def reset(self):

        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            # reset_proxy.call()
            self.reset_proxy()
            self.unpause()
        except rospy.ServiceException as e:
            print(e)
            print("/gazebo/reset_simulation service call failed")

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            # resp_pause = pause.call()
            self.unpause()
        except rospy.ServiceException as e:
            print(e)
            print("/gazebo/unpause_physics service call failed")

        # read laser data
        data = None
        while data is None:
            data = rospy.wait_for_message('/F1ROS/laser/scan', LaserScan, timeout=5)
            # print("[Laser data here]")

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except rospy.ServiceException as e:
            print(e)
            print("/gazebo/pause_physics service call failed")
        state = self.discrete_observation(data, 5)
        return state
