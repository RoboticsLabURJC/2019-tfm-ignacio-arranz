import gym
import rospy
import roslaunch
import time
import numpy as np
import cv2
import sys
import os
import random

from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from gym.utils import seeding
from cv_bridge import CvBridge, CvBridgeError

import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer


# Images size
witdh = 640
half_image_pixel = witdh/2

# Maximum distance from the line
RANGES = [300, 250, 200]
space_reward = [np.flip(np.linspace(0,1,RANGES[0])),
                np.flip(np.linspace(0,1,RANGES[1])),
                np.flip(np.linspace(0,1,RANGES[2]))]

points_in_image = len(RANGES)

last_center_line = 0


### OUTPUTS
v_lineal = [1, 2, 5]
w_angular = [-0.2, -0.1, 0, 0.2, 0.1]

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

    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "F1Cameracircuit_v0.launch")
        self.vel_pub = rospy.Publisher('/F1ROS/cmd_vel', Twist, queue_size=5)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

        # self.my_image = None

        self.reward_range = (-np.inf, np.inf)

        self._seed()

        self.last50actions = [0] * 50

        self.img_rows = 32
        self.img_cols = 32
        self.img_channels = 1


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def imageMsg2Image(self, img, cv_image):

        image = ImageF1()
        image.width = img.width
        image.height = img.height
        image.format = "RGB8"
        image.timeStamp = img.header.stamp.secs + (img.header.stamp.nsecs *1e-9)
        image.data = cv_image

        return image


    def get_line(self, image_line):

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

        line_1 = mask[260,:]
        line_2 = mask[360,:]
        line_3 = mask[450,:]

        central_1 = self.get_line(line_1)
        central_2 = self.get_line(line_2)
        central_3 = self.get_line(line_3)

        print(central_1, central_2, central_3)

        return central_1, central_2, central_3


    def callback(self, ros_data):

        print("CALLBACK!!!!: ", ros_data.height, ros_data.width)
        np_arr = np.fromstring(ros_data.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)        
        
        self.my_image = image_np

        # rospy.loginfo(rospy.get_caller_id() + "I see %s", data.data)


    def calculate_reward(self, line_1, line_2, line_3):

        reward_1 = space_reward[0][np.abs(half_image_pixel - line_1)]
        reward_2 = space_reward[1][np.abs(half_image_pixel - line_2)]
        reward_3 = space_reward[2][np.abs(half_image_pixel - line_3)]

        reward = np.divide(np.sum([reward_1, reward_2, reward_3]), points_in_image)

        #print("reward_1: {} - reward_2: {} - reward_3: {}".format(reward_1, reward_2, reward_3))

        return reward

    def calculate_values(self, line_1, line_2, line_3):
    
        ### LASER
        # min_range = 0.21
        # done = False
        # for i, item in enumerate(data.ranges):
        #     #print("-----> {}".format(data.ranges[i]))
        #     if (min_range > data.ranges[i] > 0):
        #         done = True

        done = False

        if half_image_pixel-RANGES[2] < line_3 < half_image_pixel+RANGES[2]:
            if half_image_pixel-RANGES[0] < line_1 < half_image_pixel+RANGES[0] or half_image_pixel-RANGES[1] < line_2 < half_image_pixel+RANGES[1]:
                print("In Line")
                pass
        else:
            done = True

        return done


    def step(self, action):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        '''# 21 actions
        max_ang_speed = 0.3
        ang_vel = (np.abs(action-10))*max_ang_speed*0.1   #from (-0.33 to + 0.33)

        vel_cmd = Twist()
        vel_cmd.linear.x = 0.2
        vel_cmd.angular.z = ang_vel
        self.vel_pub.publish(vel_cmd)'''

        # 5 actions
        # max_ang_speed = 1
        # ang_vel = (action-10)*max_ang_speed*0.1 #from (-0.33 to + 0.33)

        # vel_cmd = Twist()
        # vel_cmd.linear.x = 2
        # vel_cmd.angular.z = ang_vel
        # self.vel_pub.publish(vel_cmd)
 
        # print("ACTION: {}".format(action))
        # print("V LINEAL: {}".format(vel_cmd.linear.x))
        # print("W ANGULAR: {}".format(vel_cmd.angular.z))

        print("Action: {}".format(action))
        # 5 actions
        if action == 0:
            vel_cmd = Twist()
            vel_cmd.linear.x = v_lineal[2]
            vel_cmd.angular.z = w_angular[2]
            self.vel_pub.publish(vel_cmd)
        elif action == 1:
            vel_cmd = Twist()
            vel_cmd.linear.x = v_lineal[0]
            vel_cmd.angular.z = w_angular[0]
            self.vel_pub.publish(vel_cmd)
        elif action == 2:
            vel_cmd = Twist()
            vel_cmd.linear.x = v_lineal[1]
            vel_cmd.angular.z = w_angular[1]
            self.vel_pub.publish(vel_cmd)
        elif action == 3:
            vel_cmd = Twist()
            vel_cmd.linear.x = v_lineal[1]
            vel_cmd.angular.z = w_angular[3]
            self.vel_pub.publish(vel_cmd)
        elif action == 4:
            vel_cmd = Twist()
            vel_cmd.linear.x = v_lineal[0]
            vel_cmd.angular.z = w_angular[4]
            self.vel_pub.publish(vel_cmd)


        # =============
        # === IMAGE ===
        # =============
        image_data = None
        success = False
        cv_image = None
        
        # while image_data is None or success is False:
        #     try:
        #         image_data = rospy.wait_for_message('/F1ROS/cameraL/image_raw', Image, timeout=5)
        #         h = image_data.height
        #         w = image_data.width
        #         cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")

        #         # temporal fix, check image is not corrupted
        #         if not (cv_image[h//2,w//2,0]==178 and cv_image[h//2,w//2,1]==178 and cv_image[h//2,w//2,2]==178):
        #             success = True
        #         else:
        #             pass
        #             #print("/camera/rgb/image_raw ERROR, retrying")
        #     except:
        #         pass

        while image_data is None or success is False:
            image_data = rospy.wait_for_message('/F1ROS/cameraL/image_raw', Image, timeout=5)
            # Transform the image data from ROS to CVMat
            cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")
            f1_image_camera = self.imageMsg2Image(image_data, cv_image)

            line_1, line_2, line_3 = self.processed_image(f1_image_camera.data)
            
            if line_1 and line_2 and line_3:
                success = True
                

        done = self.calculate_values(line_1, line_2, line_3)

        # try:
        #     rospy.Subscriber("/F1ROS/cameraL/image_raw", Image, self.callback)
        #     # print("---------------->", self.my_image)
        #     cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")
        # except:
        #     pass

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

        # ============
        # == REWARD ==
        # ============
        # 3 actions
        if done:
            reward = self.calculate_reward(line_1, line_2, line_2)
            # if action == 0:
            #     reward = 0.8
            # elif action_sum > 45:  # L or R looping
            #     reward = -0.5
            # else:  # L or R no looping
            #     reward = 0.5
        else:
            reward = -1

        # print("detour= "+str(center_detour)+" :: reward= "+str(reward)+" ::action="+str(action))

        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        cv_image = cv2.resize(cv_image, (self.img_rows, self.img_cols))

        state = cv_image.reshape(1, 1, cv_image.shape[0], cv_image.shape[1])
            
        return state, reward, done, {}

        # test STACK 4
        #cv_image = cv_image.reshape(1, 1, cv_image.shape[0], cv_image.shape[1])
        #self.s_t = np.append(cv_image, self.s_t[:, :3, :, :], axis=1)
        #return self.s_t, reward, done, {} # observation, reward, done, info


    def reset(self):
        self.last50actions = [0] * 50 #used for looping avoidance

        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            #reset_proxy.call()
            # Reset environment. Return the robot to origina position.
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
        success = False
        cv_image = None

        

        while image_data is None or success is False:
            try:
                image_data = rospy.wait_for_message('/F1ROS/cameraL/image_raw', Image, timeout=5)
                h = image_data.height
                w = image_data.width

                cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")

                success = True
                # temporal fix, check image is not corrupted
                #if (cv_image[h//2,w//2,0]==178 and cv_image[h//2,w//2,1]==178 and cv_image[h//2,w//2,2]==178):
                if not (cv_image[320, 240, 0]==178 and cv_image[320, 240, 1]==178 and cv_image[320, 240, 2]==178):
                    success = True
                else:
                    pass
                    #print("/camera/rgb/image_raw ERROR, retrying")
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")

        '''x_t = skimage.color.rgb2gray(cv_image)
        x_t = skimage.transform.resize(x_t,(32,32))
        x_t = skimage.exposure.rescale_intensity(x_t,out_range=(0,255))'''


        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        cv_image = cv2.resize(cv_image, (self.img_rows, self.img_cols))
        #cv_image = cv_image[(self.img_rows/20):self.img_rows-(self.img_rows/20),(self.img_cols/10):self.img_cols] #crop image
        #cv_image = skimage.exposure.rescale_intensity(cv_image,out_range=(0,255))

        state = cv_image.reshape(1, 1, cv_image.shape[0], cv_image.shape[1])
        return state

        # test STACK 4
        #self.s_t = np.stack((cv_image, cv_image, cv_image, cv_image), axis=0)
        #self.s_t = self.s_t.reshape(1, self.s_t.shape[0], self.s_t.shape[1], self.s_t.shape[2])
        #return self.s_t


