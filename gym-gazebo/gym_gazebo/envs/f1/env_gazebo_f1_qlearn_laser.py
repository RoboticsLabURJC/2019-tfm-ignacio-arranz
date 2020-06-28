import gym
import rospy
import roslaunch
import time
import random
import numpy as np

from gym import utils, spaces
from gym.utils import seeding
from gym_gazebo.envs import gazebo_env
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState, SetModelState

from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from sensor_msgs.msg import LaserScan


# POSES
positions = [(0, 53.462, -41.988, 0.004, 0, 0, 1.57, -1.57),
             (1, 53.462, -8.734, 0.004, 0, 0, 1.57, -1.57),
             (2, 39.712, -30.741, 0.004, 0, 0, 1.56, 1.56),
             (3, -7.894, -39.051, 0.004, 0, 0.01, -2.021, 2.021),
             (4, 20.043, 37.130, 0.003, 0, 0.103, -1.4383, -1.4383)]

class GazeboF1QlearnLaserEnv(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "F1Lasercircuit_v0.launch")
        self.vel_pub = rospy.Publisher('/F1ROS/cmd_vel', Twist, queue_size=5)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.action_space = spaces.Discrete(5)  # F, L, R
        self.reward_range = (-np.inf, np.inf)
        self.position = None
        self._seed()

    def render(self, mode='human'):
        pass

    def _gazebo_pause(self):
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except rospy.ServiceException as e:
            print("/gazebo/pause_physics service call failed: {}".format(e))

    def _gazebo_unpause(self):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except rospy.ServiceException as e:
            print(e)
            print("/gazebo/unpause_physics service call failed")

    def _gazebo_reset(self):
        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            # reset_proxy.call()
            self.reset_proxy()
            self.unpause()
        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed: {}".format(e))

    @staticmethod
    def set_new_pose(new_pos):
        """
        (pos_number, pose_x, pose_y, pose_z, or_x, or_y, or_z, or_z)
        """
        pos_number = positions[0]

        state = ModelState()
        state.model_name = "f1_renault"
        state.pose.position.x = positions[new_pos][1]
        state.pose.position.y = positions[new_pos][2]
        state.pose.position.z = positions[new_pos][3]
        state.pose.orientation.x = positions[new_pos][4]
        state.pose.orientation.y = positions[new_pos][5]
        state.pose.orientation.z = positions[new_pos][6]
        state.pose.orientation.w = positions[new_pos][7]

        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        except rospy.ServiceException as e:
            print("Service call failed: {}".format(e))
        return pos_number

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
        self._gazebo_unpause()

        vel_cmd = Twist()
        if action == 0:  # FORWARD 1
            vel_cmd.linear.x = 3
            vel_cmd.angular.z = 0.0
        if action == 1:  # FORWARD 2
            vel_cmd.linear.x = 6
            vel_cmd.angular.z = 0.0
        elif action == 2:  # LEFT 1
            vel_cmd.linear.x = 3  # 0.05
            vel_cmd.angular.z = 1  # 0.3
        elif action == 3:  # RIGHT 1
            vel_cmd.linear.x = 3  # 0.05
            vel_cmd.angular.z = -1  # -0.3
        elif action == 4:  # LEFT 2
            vel_cmd.linear.x = 4  # 0.05
            vel_cmd.angular.z = 4  # 0.3
        elif action == 5:  # RIGHT 2
            vel_cmd.linear.x = 4  # 0.05
            vel_cmd.angular.z = -4  # -0.3
        self.vel_pub.publish(vel_cmd)

        laser_data = None
        success = False
        while laser_data is None or not success:
            try:
                laser_data = rospy.wait_for_message('/F1ROS/laser/scan', LaserScan, timeout=5)
            finally:
                success = True

        self._gazebo_pause()

        state, _ = self.discrete_observation(laser_data, 5)

        laser_len = len(laser_data.ranges)
        left_sum = sum(laser_data.ranges[laser_len - (laser_len / 5):laser_len - (laser_len / 10)])  # 80-90
        right_sum = sum(laser_data.ranges[(laser_len / 10):(laser_len / 5)])  # 10-20
        center_detour = (right_sum - left_sum) / 5

        done = False
        if abs(center_detour) > 4:
            done = True
        # print("center: {}".format(center_detour))

        #############
        # 3 actions
        #############
        if not done:
            if abs(center_detour) < 4:
                 reward = 5
            elif abs(center_detour < 2) and action == 1:
                reward = 10
            else:  # L or R no looping
                reward = 2
        else:
            reward = -200


        return state, reward, done, {}

    def reset(self):

        # ========
        # = POSE =
        # ========
        # pos = random.choice(list(enumerate(positions)))[0]
        # print(self.position)

        self._gazebo_reset()
        self._gazebo_unpause()

        # read laser data
        laser_data = None
        success = False
        while laser_data is None or not success:
            try:
                laser_data = rospy.wait_for_message('/F1ROS/laser/scan', LaserScan, timeout=5)
            finally:
                success = True

        self._gazebo_pause()

        state = self.discrete_observation(laser_data, 5)

        return state
