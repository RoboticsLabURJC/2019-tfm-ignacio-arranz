import os
import random
import sys
import time

import cv2
import gym
import numpy as np
import roslaunch
import rospkg
import rospy
import skimage as skimage

from cv_bridge import CvBridge, CvBridgeError
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState, SetModelState
from geometry_msgs.msg import Twist
from gym import spaces, utils
from gym.utils import seeding
from sensor_msgs.msg import Image, LaserScan
from skimage import color, exposure, transform
from skimage.transform import rotate
from skimage.viewer import ImageViewer
from std_srvs.srv import Empty

from gym_gazebo.envs import gazebo_env

# Images size
witdh = 640
center_image = witdh/2

# Coord X ROW
x_row = [260, 360, 450]
# Maximum distance from the line
RANGES = [300, 280, 250]  # Line 1, 2 and 3

RESET_RANGE = [-40, 40]

# Deprecated?
space_reward = np.flip(np.linspace(0, 1, 300))

last_center_line = 0

### OUTPUTS
v_lineal = [3, 8, 15]
w_angular = [-1, -0.6, 0, 1, 0.6]

### POSES
positions = [(53.462, -38.9884, 0.004, 0, 0, 1.57, -1.57),
             (53.462, -10.734, 0.004, 0, 0, 1.57, -1.57)]

class ImageF1:
    def __init__(self):
        self.height = 3  # Image height [pixels]
        self.width = 3  # Image width [pixels]
        self.timeStamp = 0 # Time stamp [s] */
        self.format = "" # Image format string (RGB8, BGR,...)
        self.data = np.zeros((self.height, self.width, 3), np.uint8) # The image data itself
        self.data.shape = self.height, self.width, 3

    def __str__(self):
        s = "Image: {\n   height: " + str(self.height) + "\n   width: " + str(self.width)
        s = s + "\n   format: " + self.format + "\n   timeStamp: " + str(self.timeStamp) 
        s = s + "\n   data: " + str(self.data) + "\n}"


class GazeboF1CameraEnv(gazebo_env.GazeboEnv):
    """
    Description:
        A Formula 1 car has to complete one lap of a circuit following a red line painted on the ground. Initially it will
        not use the complete information of the image but some coordinates that refer to the error made with respect to the
        center of the line.
    Source:
        Master's final project at Universidad Rey Juan Carlos. RoboticsLab Urjc. JdeRobot. Author: Ignacio Arranz
    Observation: 
        Type: Array
        Num	Observation               Min   Max
        ----------------------------------------
        0	Vel. Lineal (m/s)         1     10
        1	Vel. Angular (rad/seg)   -2     2
        2	Error 1                  -300   300
        3	Error 2                  -280   280
        4   Error 3                  -250   250
    Actions:
        Type: Dict
        Num	Action
        ----------
        0:   -2
        1:   -1
        2:    0
        3:    1
        4:    2
    Reward:
        The reward depends on the set of the 3 errors. As long as the lowest point is within range, there will still
        be steps. If errors 1 and 2 fall outside the range, a joint but weighted reward will be posted, always giving
        more importance to the lower one.
    Starting State:
        The observations will start from different points in order to prevent you from performing the same actions
        initially. This variability will enrich the set of state/actions.
    Episode Termination:
        The episode ends when the lower error is outside the established range (see table in the observation space).
    """


    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "F1Cameracircuit_v0.launch")
        self.vel_pub = rospy.Publisher('/F1ROS/cmd_vel', Twist, queue_size=5)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

        self.state_msg = ModelState()
        self.set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
        self.state_msg.model_name = 'f1_renault'        
        self.reward_range = (-np.inf, np.inf)
        self._seed()
        
        self.last50actions = [0] * 50
        
        self.img_rows = 32
        self.img_cols = 32
        self.img_channels = 1

        self.action_space = self._generate_simple_action_space()


    def _generate_simple_action_space(self):
        actions = 5

        max_ang_speed = -4

        action_space = {}

        for action in range(actions):    
            if action > actions/2:
                diff = action - round(actions/2)
            vel_ang = round((action - actions/2) * max_ang_speed * 0.1 , 2) # from (-1 to + 1)
            action_space[action] = vel_ang
    
        return action_space


    def _generate_action_space(self):
        actions = 21

        max_ang_speed = 1.5
        min_lin_speed = 2
        max_lin_speed = 12

        action_space_dict = {}

        for action in range(actions):    
            if action > actions/2:
                diff = action - round(actions/2)
                vel_lin = max_lin_speed - diff # from (3 to 15)
            else:
                vel_lin = action + min_lin_speed # from (3 to 15)
            vel_ang = round((action - actions/2) * max_ang_speed * 0.1 , 2) # from (-1 to + 1)
            action_space_dict[action] = (vel_lin, vel_ang)
            # print("Action: {} - V: {} - W: {}".format(action, vel_lin, vel_ang))
        # print(action_space_dict)
    
        return action_space_dict


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def set_new_pose(self, new_pos):

        # (pose_x, pose_y, pose_z, or_x, or_y, or_z, or_z)
        self.state_msg.pose.position.x = new_pos[0]
        self.state_msg.pose.position.y = new_pos[1]
        self.state_msg.pose.position.z = new_pos[2]
        self.state_msg.pose.orientation.x = new_pos[3]
        self.state_msg.pose.orientation.y = new_pos[4]
        self.state_msg.pose.orientation.z = new_pos[5]
        self.state_msg.pose.orientation.w = new_pos[6]

        rospy.wait_for_service('/gazebo/set_model_state')

        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            # resp = set_state(self.state_msg)
        except rospy.ServiceException, e:
            print("Service call failed: %s") % e


    def imageMsg2Image(self, img, cv_image):

        image = ImageF1()
        image.width = img.width
        image.height = img.height
        image.format = "RGB8"
        image.timeStamp = img.header.stamp.secs + (img.header.stamp.nsecs *1e-9)
        image.data = cv_image

        return image


    def get_center(self, image_line):

        try:
            coords = np.divide(np.max(np.nonzero(image_line)) - np.min(np.nonzero(image_line)), 2)
            coords = np.min(np.nonzero(image_line)) + coords
        except:
            coords = -1

        return coords


    def processed_image(self, img):
        
        """
        Conver img to HSV. Get the image processed. Get 3 lines from the image.

        :parameters: input image 640x480
        :return: x, y, z: 3 coordinates
        """

        img_proc = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        line_pre_proc = cv2.inRange(img_proc, (0, 30, 30), (0, 255, 200))

        #gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(line_pre_proc, 240, 255, cv2.THRESH_BINARY)

        line_1 = mask[x_row[0],:]
        line_2 = mask[x_row[1],:]
        line_3 = mask[x_row[2],:]

        central_1 = self.get_center(line_1)
        central_2 = self.get_center(line_2)
        central_3 = self.get_center(line_3)

        #print(central_1, central_2, central_3)

        return central_1, central_2, central_3


    def callback(self, ros_data):

        print("CALLBACK!!!!: ", ros_data.height, ros_data.width)
        np_arr = np.fromstring(ros_data.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)        
        
        self.my_image = image_np
        # rospy.loginfo(rospy.get_caller_id() + "I see %s", data.data)


    def calculate_error(self, point_1, point_2, point_3):

        error_1 = abs(center_image - point_1)
        error_2 = abs(center_image - point_2)
        error_3 = abs(center_image - point_3)

        return error_1, error_2, error_3
        

    def calculate_reward(self, error_1, error_2, error_3):

        global center_image
        ALPHA = 0
        BETA = 0
        GAMMA = 1

        if error_1 > RANGES[0] and error_2 > RANGES[1]:
            ALPHA = 0.1
            BETA = 0.2
            GAMMA = 0.7
        elif error_1 > RANGES[0]:
            ALPHA = 0.1
            BETA = 0
            GAMMA = 0.9
        elif error_2 > RANGES[1]:
            ALPHA = 0
            BETA = 0.1
            GAMMA = 0.9

        d = ALPHA * np.true_divide(error_1, center_image) + \
            BETA  * np.true_divide(error_2, center_image) + \
            GAMMA * np.true_divide(error_3, center_image)

        reward = np.round(np.exp(-d), 4)

        return reward


    def is_game_over(self, point_1, point_2, point_3):

        done = False
    
        if center_image-RANGES[2] < point_3 < center_image+RANGES[2]:
            if center_image-RANGES[0] < point_1 < center_image+RANGES[0] or center_image-RANGES[1] < point_2 < center_image+RANGES[1]:
                pass # In Line
        else:
            done = True

        return done


    def step(self, action):

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        # =============
        # == ACTIONS == - 5 actions
        # =============
        vel_cmd = Twist()
        vel_cmd.linear.x = 3 # self.action_space[action][0]
        vel_cmd.angular.z = self.action_space[action] # [1]
        self.vel_pub.publish(vel_cmd)
        # print("Action: {} - V_Lineal: {} - W_Angular: {}".format(action, vel_cmd.linear.x, vel_cmd.angular.z))

        # ===========
        # == IMAGE == 
        # ===========
        image_data = None
        success = False
        cv_image = None
        while image_data is None or success is False:
            image_data = rospy.wait_for_message('/F1ROS/cameraL/image_raw', Image, timeout=5)
            # Transform the image data from ROS to CVMat
            cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")
            f1_image_camera = self.imageMsg2Image(image_data, cv_image)

            if f1_image_camera:
                success = True


        point_1, point_2, point_3 = self.processed_image(f1_image_camera.data)
        
        ### DONE
        done = self.is_game_over(point_1, point_2, point_3)

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")


        self.last50actions.pop(0) #remove oldest
        if action == 0:
            self.last50actions.append(0)
        else:
            self.last50actions.append(1)

        action_sum = sum(self.last50actions)

        # =====================
        # == CALCULATE ERROR ==
        # =====================
        error_1, error_2, error_3 = self.calculate_error(point_1, point_2, point_3)

        # ============
        # == REWARD ==
        # ============
        if not done:
            reward = self.calculate_reward(error_1, error_2, error_3)
        else:
            reward = -1

        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        cv_image = cv2.resize(cv_image, (self.img_rows, self.img_cols))
        observation = cv_image.reshape(1, 1, cv_image.shape[0], cv_image.shape[1])
        
        #info = [vel_cmd.linear.x, vel_cmd.angular.z, error_1, error_2, error_3]
        
        # OpenAI standard return: observation, reward, done, info
        return observation, reward, done, {}

        # test STACK 4
        #cv_image = cv_image.reshape(1, 1, cv_image.shape[0], cv_image.shape[1])
        #self.s_t = np.append(cv_image, self.s_t[:, :3, :, :], axis=1)
        #return self.s_t, reward, done, {} # observation, reward, done, info


    def reset(self):
        
        self.last50actions = [0] * 50  # used for looping avoidance

        # ========
        # = POSE =
        # ========
        pos = 0
        # pos = random.choice(positions)
        # self.set_new_pose(pos)        
        # print(pos)
        
        # =========
        # = RESET =
        # =========
        # Resets the state of the environment and returns an initial observation.
        time.sleep(0.05)
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            #reset_proxy.call()
            # Reset environment. Return the robot to original position.
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("/gazebo/reset_simulation service call failed")

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            #resp_pause = pause.call()
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")


        image_data = None
        cv_image = None
        success = False
        while image_data is None or success is False:
            image_data = rospy.wait_for_message('/F1ROS/cameraL/image_raw', Image, timeout=5)
            # Transform the image data from ROS to CVMat
            cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")
            f1_image_camera = self.imageMsg2Image(image_data, cv_image)
            
            if f1_image_camera:
                success = True
            else:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")


        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        cv_image = cv2.resize(cv_image, (self.img_rows, self.img_cols))
        #cv_image = cv_image[(self.img_rows/20):self.img_rows-(self.img_rows/20),(self.img_cols/10):self.img_cols] #crop image
        #cv_image = skimage.exposure.rescale_intensity(cv_image,out_range=(0,255))

        state = cv_image.reshape(1, 1, cv_image.shape[0], cv_image.shape[1])
        return state, pos

        # test STACK 4
        #self.s_t = np.stack((cv_image, cv_image, cv_image, cv_image), axis=0)
        #self.s_t = self.s_t.reshape(1, self.s_t.shape[0], self.s_t.shape[1], self.s_t.shape[2])
        #return self.s_t
