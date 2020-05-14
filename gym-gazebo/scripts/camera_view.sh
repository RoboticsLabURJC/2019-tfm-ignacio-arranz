#!/bin/bash

TOPIC=""

if [ "$1" == "f1" ]; then
    TOPIC="/F1ROS/cameraL/image_raw"
elif [ "$1" == "turtlebot" ]; then
    TOPIC="/camera/rgb/image_raw"
else
    printf "\n[ERROR] - No robot selected\n\n"
    exit 1
fi

rosrun image_view image_view image:=$TOPIC
