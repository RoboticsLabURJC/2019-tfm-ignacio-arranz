import os
import cv2
import sys
import rospy
import rosbag
import signal
import datetime
import argparse
import numpy as np

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from gym_pyxis.envs.gazebo.turtlebot import turtlebot_utils

exit = False
def signal_handler(signal_number, frame):
    global exit
    exit = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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

                # turtlebot_utils.get_robot_position_respect_road(np.copy(frame), debug=True)
                if old_frame is None:
                    old_frame = frame
                    continue

                turtlebot_utils.estimate_movement(np.copy(frame), np.copy(old_frame))
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
        bag = rosbag.Bag('turtlebot3_training_{}.bag'.format(t.strftime('%Y%m%d_%H%M%S')), 'w')

        try:
            timeout = 5
            rospy.init_node('training_logger', anonymous=True)
            while not exit:

                image_data = rospy.wait_for_message('/camera/rgb/image_raw', Image, timeout=timeout)
                if image_data is not None:
                    bag.write('/camera/rgb/image_raw', image_data)

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

        with rosbag.Bag(args.read) as bag:
            for topic, msg, t in bag.read_messages(topics=['/camera/rgb/image_raw']):
                if topic == '/camera/rgb/image_raw':
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

            if output_video is not None:
                output_video.release()
