import rospy
import time
import numpy as np
import random
import cv2

from gym import spaces
from cv_bridge import CvBridge

from gym_gazebo.envs import gazebo_env
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState

from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from sensor_msgs.msg import Image

from gym.utils import seeding
from agents.f1.settings import actions, gazebo_positions
from agents.f1.settings import telemetry, x_row, ranges, center_image, width, height, telemetry_mask


font = cv2.FONT_HERSHEY_COMPLEX


class ImageF1:
    def __init__(self):
        self.height = 3  # Image height [pixels]
        self.width = 3  # Image width [pixels]
        self.timeStamp = 0  # Time stamp [s] */
        self.format = ""  # Image format string (RGB8, BGR,...)
        self.data = np.zeros((self.height, self.width, 3), np.uint8)  # The image data itself
        self.data.shape = self.height, self.width, 3

    def __str__(self):
        s = "Image: {\n   height: " + str(self.height) + "\n   width: " + str(self.width)
        s = s + "\n   format: " + self.format + "\n   timeStamp: " + str(self.timeStamp)
        return s + "\n   data: " + str(self.data) + "\n}"


class GazeboF1QlearnCameraEnv(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        # gazebo_env.GazeboEnv.__init__(self, "F1Cameracircuit_v0.launch")
        gazebo_env.GazeboEnv.__init__(self, "f1_1_nurburgrinlineROS.launch")
        self.vel_pub = rospy.Publisher('/F1ROS/cmd_vel', Twist, queue_size=5)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.action_space = spaces.Discrete(len(actions))  # actions  # spaces.Discrete(3)  # F,L,R
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

    @staticmethod
    def all_same(items):
        return all(x == items[0] for x in items)

    def _gazebo_reset(self):
        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            # reset_proxy.call()
            self.reset_proxy()
            self.unpause()
        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed: {}".format(e))

    def set_new_pose(self):
        """
        (pos_number, pose_x, pose_y, pose_z, or_x, or_y, or_z, or_z)
        """
        pos = random.choice(list(enumerate(gazebo_positions)))[0]
        self.position = pos

        pos_number = gazebo_positions[0]

        state = ModelState()
        state.model_name = "f1_renault"
        state.pose.position.x = gazebo_positions[pos][1]
        state.pose.position.y = gazebo_positions[pos][2]
        state.pose.position.z = gazebo_positions[pos][3]
        state.pose.orientation.x = gazebo_positions[pos][4]
        state.pose.orientation.y = gazebo_positions[pos][5]
        state.pose.orientation.z = gazebo_positions[pos][6]
        state.pose.orientation.w = gazebo_positions[pos][7]

        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            set_state(state)
        except rospy.ServiceException as e:
            print("Service call failed: {}".format(e))
        return pos_number

    @staticmethod
    def image_msg_to_image(img, cv_image):

        image = ImageF1()
        image.width = img.width
        image.height = img.height
        image.format = "RGB8"
        image.timeStamp = img.header.stamp.secs + (img.header.stamp.nsecs * 1e-9)
        image.data = cv_image

        return image

    @staticmethod
    def get_center(lines):

        try:
            point = np.divide(np.max(np.nonzero(lines)) - np.min(np.nonzero(lines)), 2)
            point = np.min(np.nonzero(lines)) + point
        except:
            point = 9

        return point

    @staticmethod
    def calculate_reward(error):

        d = np.true_divide(error, center_image)
        reward = np.round(np.exp(-d), 4)

        return reward

    def processed_image(self, img):
        """
        Convert img to HSV. Get the image processed. Get 3 lines from the image.

        :parameters: input image 640x480
        :return: x, y, z: 3 coordinates
        """

        img_proc = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        line_pre_proc = cv2.inRange(img_proc, (0, 30, 30), (0, 255, 200))
        # gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(line_pre_proc, 240, 255, cv2.THRESH_BINARY)

        lines = [mask[x_row[idx], :] for idx, x in enumerate(x_row)]
        centrals = map(self.get_center, lines)

        # if centrals[-1] == 9:
        #     centrals[-1] = center_image

        if telemetry_mask:
            mask_points = np.zeros((height, width), dtype=np.uint8)
            for idx, point in enumerate(centrals):
                # mask_points[x_row[idx], centrals[idx]] = 255
                cv2.line(mask_points, (point, x_row[idx]), (point, x_row[idx]), (255, 255, 255), thickness=3)

            cv2.imshow("MASK", mask_points[240:])
            cv2.waitKey(3)

        return centrals

    @staticmethod
    def calculate_observation(state):

        normalize = 40

        final_state = []
        for _, x in enumerate(state):
            final_state.append(abs((center_image - x) / normalize) + 1)

        return final_state

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @staticmethod
    def show_telemetry(img, points, action, reward):
        count = 0
        for idx, point in enumerate(points):
            cv2.line(img, (320, x_row[idx]), (320, x_row[idx]), (255, 255, 0), thickness=5)
            # cv2.line(img, (center_image, x_row[idx]), (point, x_row[idx]), (255, 255, 255), thickness=2)
            cv2.putText(img, str("err{}: {}".format(idx+1, center_image - point)), (18, 340 + count), font, 0.4,
                        (255, 255, 255), 1)
            count += 20
        cv2.putText(img, str("action: {}".format(action)), (18, 280), font, 0.4, (255, 255, 255), 1)
        cv2.putText(img, str("reward: {}".format(reward)), (18, 320), font, 0.4, (255, 255, 255), 1)

        cv2.imshow("Image window", img[240:])
        cv2.waitKey(3)

    def step(self, action):

        self._gazebo_unpause()

        vel_cmd = Twist()
        vel_cmd.linear.x = actions[action][0]
        vel_cmd.angular.z = actions[action][1]
        self.vel_pub.publish(vel_cmd)

        # Get camera info
        image_data = None
        f1_image_camera = None
        while image_data is None:
            image_data = rospy.wait_for_message('/F1ROS/cameraL/image_raw', Image, timeout=5)
            # Transform the image data from ROS to CVMat
            cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")
            f1_image_camera = self.image_msg_to_image(image_data, cv_image)

        self._gazebo_pause()

        points = self.processed_image(f1_image_camera.data)
        state = self.calculate_observation(points)

        center = float(center_image - points[-1]) / (float(width) // 2)

        done = False
        center = abs(center)

        if center > 0.8:
            done = True
        if not done:
            if 0 <= center <= 0.2:
                reward = 10
            elif 0.2 < center <= 0.4:
                reward = 2
            else:
                reward = 1
        else:
            reward = -100

        if telemetry:
            print("center: {} - actions: {} - reward: {}".format(center, action, reward))
            # self.show_telemetry(f1_image_camera.data, points, action, reward)

        return state, reward, done, {}

    def reset(self):
        # === POSE ===
        self.set_new_pose()
        # self._gazebo_reset()

        self._gazebo_unpause()

        # Get camera info
        image_data = None
        f1_image_camera = None
        success = False
        while image_data is None or success is False:
            image_data = rospy.wait_for_message('/F1ROS/cameraL/image_raw', Image, timeout=5)
            cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")
            f1_image_camera = self.image_msg_to_image(image_data, cv_image)
            if f1_image_camera:
                success = True

        points = self.processed_image(f1_image_camera.data)
        state = self.calculate_observation(points)
        # reset_state = (state, False)

        self._gazebo_pause()

        return state
