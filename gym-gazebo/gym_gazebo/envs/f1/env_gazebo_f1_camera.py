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
mid = 320

# Maximum distance from the line
RANGES = [200, 100, 70]

last_center_line = 0


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


    def processed_image(self, img):
        
        """
        Conver img to HSV. Get the image processed. Get 3 lines from the image.

        :parameters: input image 640x480
        :return: x, y, z: 3 coordinates
        """

        # img = img[220:]
        # img_proc = cv2.cvtColor(img[220:], cv2.COLOR_BGR2HSV)
        
        img_proc = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        mask = cv2.inRange(img_proc, (0, 30, 30), (0, 255, 200))

        cv2.imwrite('/home/nachoaz/Desktop/myImage.png', mask)

        wall = img[12][320][0]
        mask_1 = mask[30,:]
        mask_2 = mask[110,:]
        mask_3 = mask[210,:]
        base = mask[250,:]

        print(mask_1)
        print("------------------------------------------------")
        print(np.nonzero(mask_2))
        print("------------------------------------------------")
        print(np.max(np.nonzero(mask_1)))
        print("------------------------------------------------")
        print(np.max(np.nonzero(mask_1))- np.min(np.nonzero(mask_1)))
        print("------------------------------------------------")

        line_1 = np.divide(np.max(np.nonzero(mask_1)) - np.min(np.nonzero(mask_1)), 2)
        line_1 = np.min(np.nonzero(mask_1)) + line_1
        line_2 = np.divide(np.max(np.nonzero(mask_2)) - np.min(np.nonzero(mask_2)), 2)
        line_2 = np.min(np.nonzero(mask_2)) + line_2
        line_3 = np.divide(np.max(np.nonzero(mask_3)) - np.min(np.nonzero(mask_3)), 2)
        line_3 = np.min(np.nonzero(mask_3)) + line_3

        print(line_1, line_2, line_3)

        return line_1, line_2, line_3




    def callback(self, ros_data):

        print("CALLBACK!!!!: ", ros_data.height, ros_data.width)
        np_arr = np.fromstring(ros_data.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)        
        
        self.my_image = image_np

        # rospy.loginfo(rospy.get_caller_id() + "I see %s", data.data)


    def calculate_observation(self, image):
    
        ### LASER
        # min_range = 0.21
        # done = False
        # for i, item in enumerate(data.ranges):
        #     #print("-----> {}".format(data.ranges[i]))
        #     if (min_range > data.ranges[i] > 0):
        #         done = True
        done = False
        print("=====================================================0")
        #cv2.imwrite('/home/nachoaz/Desktop/myImage.png', image)
        x, y, z = self.processed_image(image)

        print("\n\n---------------------------> {}\n\n".format(x,y,z))        
        
        if not range[0] < x < -range[0] or not range[0] < y < -range[0] or not range[0] < z < -range[0]:
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
        ang_vel = (action-10)*max_ang_speed*0.1 #from (-0.33 to + 0.33)

        vel_cmd = Twist()
        vel_cmd.linear.x = 0.2
        vel_cmd.angular.z = ang_vel
        self.vel_pub.publish(vel_cmd)'''

        # 3 actions
        if action == 0:  # FORWARD
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.2  # Default 0.2 - mini test = 2
            vel_cmd.angular.z = 0.0
            self.vel_pub.publish(vel_cmd)
        elif action == 1:  # LEFT
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.05
            vel_cmd.angular.z = 0.2
            self.vel_pub.publish(vel_cmd)
        elif action == 2:  # RIGHT
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.05
            vel_cmd.angular.z = -0.2
            self.vel_pub.publish(vel_cmd)


        # =============
        # === LASER ===
        # =============
        # data = None
        # while data is None:
        #     try:
        #         data = rospy.wait_for_message('/F1ROS/laser/scan', LaserScan, timeout=5)
        #         print("TENGO INFORMACION DEL LASER")
        #     except:
        #         pass

        # done = self.calculate_observation(data)

        # =============
        # === IMAGE ===
        # =============
        image_data = None
        success = False
        cv_image = None
        while image_data is None or success is False:
            try:
                image_data = rospy.wait_for_message('/F1ROS/cameraL/image_raw', Image, timeout=5)
                h = image_data.height
                w = image_data.width
                cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")
                #cv_image = processed_image(image_data)
                # temporal fix, check image is not corrupted
                if not (cv_image[h//2,w//2,0]==178 and cv_image[h//2,w//2,1]==178 and cv_image[h//2,w//2,2]==178):
                    success = True
                else:
                    pass
                    #print("/camera/rgb/image_raw ERROR, retrying")
            except:
                pass


        done = self.calculate_observation(cv_image)

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


        # =============
        # === LASER ===
        # =============
        '''# 21 actions
        if not done:
            # Straight reward = 5, Max angle reward = 0.5
            reward = round(15*(max_ang_speed - abs(ang_vel) +0.0335), 2)
            # print ("Action : "+str(action)+" Ang_vel : "+str(ang_vel)+" reward="+str(reward))

            if action_sum > 45: #L or R looping
                #print("90 percent of the last 50 actions were turns. LOOPING")
                reward = -5
        else:
            reward = -200'''

        # Add center of the track reward
        # len(data.ranges) = 100
        #laser_len = len(data.ranges)
        #left_sum = sum(data.ranges[laser_len-(laser_len/5):laser_len-(laser_len/10)]) #80-90
        #right_sum = sum(data.ranges[(laser_len/10):(laser_len/5)]) #10-20

        #center_detour = abs(right_sum - left_sum)/5

        # ============
        # == REWARD ==
        # ============
        # 3 actions
        if not done:
            if action == 0:
                print("RECOMPENSA PARA LA ACCION 0")
            elif action_sum > 45: # L or R looping
                print("RECOMPENSA PARA LA ACCION 1")
                #reward = -0.5
            else: # L or R no looping
                print("RECOMPENSA PARA LA ACCION 2")
        else:
            reward = -1
        
        # print("detour= "+str(center_detour)+" :: reward= "+str(reward)+" ::action="+str(action))

        '''x_t = skimage.color.rgb2gray(cv_image)
        x_t = skimage.transform.resize(x_t,(32,32))
        x_t = skimage.exposure.rescale_intensity(x_t,out_range=(0,255))'''
        # state = None
        # if cv_image:
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        cv_image = cv2.resize(cv_image, (self.img_rows, self.img_cols))
        #cv_image = cv_image[(self.img_rows/20):self.img_rows-(self.img_rows/20),(self.img_cols/10):self.img_cols] #crop image
        #cv_image = skimage.exposure.rescale_intensity(cv_image,out_range=(0,255))
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

                print(cv_image[400, 240])

                # temporal fix, check image is not corrupted
                # if (cv_image[h//2,w//2,0]==178 and cv_image[h//2,w//2,1]==178 and cv_image[h//2,w//2,2]==178):
                if not (cv_image[320, 240, 0]==178 and cv_image[320, 240, 1]==178 and cv_image[320, 240, 2]==178):
                    print("SUCCESS")
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


