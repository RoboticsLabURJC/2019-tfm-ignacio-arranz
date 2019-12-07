import os
import cv2
import sys
import math
import rospy
import rosbag
import signal
import datetime
import argparse
import numpy as np

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from gym_pyxis.envs.gazebo.f1 import f1_utils

from tf.transformations import euler_from_quaternion

exit = False
def signal_handler(signal_number, frame):
    global exit
    exit = True


def plot_path(odometry):
    mpl.rcParams['legend.fontsize'] = 10

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    x = []
    y = []
    z = []
    for msg in odometry:
        x.append(msg.pose.pose.position.x)
        y.append(msg.pose.pose.position.y)
        z.append(msg.pose.pose.position.z)

    x = [v - max(x) for v in x]
    y = [v - max(y) for v in y]
    z = [v - max(z) for v in z]

    xmax = max(x)
    if xmax == 0:
        xmax = abs(max(x) - min(x))

    ax.set_xlim([min(x) * 2 , xmax * 2])
    ax.set_ylim([min(y) * 0.5 ,  0.5])
    ax.plot(x, y, z, label='path')
    ax.legend()

    plt.figure()
    plt.plot(x,y)
    plt.ylim(0.0, -26)
    plt.xlim(-0.15, 0.10)


def plot_orientation(odometry):

    x = []
    roll = []
    pitch = []
    yaw = []
    for msg in odometry:
        x.append(msg.header.stamp.to_sec())
        euler_orientation = euler_from_quaternion((msg.pose.pose.orientation.x,
                                                   msg.pose.pose.orientation.y,
                                                   msg.pose.pose.orientation.z,
                                                   msg.pose.pose.orientation.w))
        roll.append(euler_orientation[0])
        pitch.append(euler_orientation[1])
        yaw.append(euler_orientation[2])

    norm_yaw = [x - max(yaw) for x in yaw]
    norm_roll = [x - max(roll) for x in roll]
    norm_pitch = [x - max(pitch) for x in pitch]

    max_y = max([max(norm_roll), max(norm_pitch), max(norm_yaw)])
    min_y = min([min(norm_roll), min(norm_pitch), min(norm_yaw)])

    if max_y == 0:
        max_y = 0.1

    if min_y == 0:
        min_y = -0.1

    plt.figure()
    plt.ylim(min_y * 2, max_y * 0.5)
    plt.plot(x, norm_roll, 'b', label='roll')
    plt.plot(x, norm_pitch, 'r', label='pitch')
    plt.plot(x, norm_yaw, 'g', label='yaw')
    plt.gca().legend(('roll', 'pitch', 'yaw'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", choices=['images', 'odom', 'both'], default='both', required=False)
    parser.add_argument("--read", type=str, help="Path to .bag file to read")
    parser.add_argument("--debug_video", type=str, help="Path to .bag file to read")
    args = parser.parse_args()

    if args.debug_video is not None:
        cap = cv2.VideoCapture(args.debug_video)

        if not cap.isOpened():
            print('Error opening video file')
            sys.exit(-1)

        old_frame = None
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # cv2.imshow('input', frame)

                # f1.get_robot_position_respect_road(np.copy(frame), debug=True)
                if old_frame is None:
                    old_frame = frame
                    continue

                f1_utils.estimate_movement(np.copy(frame), np.copy(old_frame))
                old_frame = frame

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()
        sys.exit(0)

    if args.read is None:
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGQUIT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGABRT, signal_handler)

        t = datetime.datetime.now()
        bag = rosbag.Bag('f1_training_{}.bag'.format(t.strftime('%Y%m%d_%H%M%S')), 'w')

        try:
            timeout = 5
            rospy.init_node('training_logger', anonymous=True)
            while not exit:

                if args.capture == 'both' or args.capture == 'images':
                    image_data = rospy.wait_for_message('/F1ROS/cameraL/image_raw', Image, timeout=timeout)
                    if image_data is not None:
                        bag.write('/F1ROS/cameraL/image_raw', image_data)

                if args.capture == 'both' or args.capture == 'odom':
                    odom_data = rospy.wait_for_message('/F1ROS/odom', Odometry, timeout=timeout)
                    if odom_data is not None:
                        bag.write('/F1ROS/odom', odom_data)

        except Exception as e:
            print('Exception raised: {}'.format(e))
        finally:
            bag.close()
    else:
        bag = rosbag.Bag(args.read)
        bridge = CvBridge()
        output_video_name = os.path.join(os.path.dirname(args.read), os.path.splitext(os.path.basename(args.read))[0])
        output_video = None
        framerate = 10

        odometry = []
        with rosbag.Bag(args.read) as bag:
            for topic, msg, t in bag.read_messages():
                if topic == '/F1ROS/cameraL/image_raw':
                    try:
                        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
                        if output_video is None:
                            output_video = cv2.VideoWriter('{}.avi'.format(output_video_name),
                                                           cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), framerate,
                                                           (cv_image.shape[1], cv_image.shape[0]))
                        output_video.write(cv_image)
                    except CvBridgeError as e:
                        print(e)

                    timestr = "%.6f" % msg.header.stamp.to_sec()
                    # print(timestr)
                    # image_name = str(save_dir)+"/"+timestr+"_left"+".pgm"
                elif topic == '/F1ROS/odom':
                    odometry.append(msg)

            if len(odometry) > 0:
                plot_path(odometry)
                plot_orientation(odometry)
                plt.show()

            if output_video is not None:
                output_video.release()
