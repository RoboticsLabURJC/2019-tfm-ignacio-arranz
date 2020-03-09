#!/usr/bin/python
#-*- coding: utf-8 -*-

import threading
import time
import math
import cv2
import numpy as np
from datetime import datetime

from interfaces.motors import PublisherMotors

from follow_line import FollowLine
from printer import printImage


# Show text into an image
font = cv2.FONT_HERSHEY_SIMPLEX

witdh = 640
mid = 320

last_center_line = 0

# Constantes de giro - kp más alta corrige más
kp = 0.02       ## valores para 20 m/s --> 0.019
kd = 0.012        ## valores para 20 m/s --> 0.011
last_error = 0

# Constantes de Velocidad
kpv = 0.01    ## valores para 20 m/s --> 0.09
kdv = 0.03   ## valores para 20 m/s --> 0.003
vel_max = 20  ## probado con 20 m/s
last_vel = 0    



def processed_image(img):
    
    """
    Conver img to HSV. Get the image processed. Get 3 lines from the image.
    Se calcula el centro de la línea roja en la imagen segmentada (valores a 255,255,255).

    """

    img = img[220:]
    img_proc = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    img_proc = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img_proc, (0, 30, 30), (0, 255, 200))
    wall = img[12][320][0]
    mask_1 = mask[30,:]
    mask_2 = mask[110,:]
    mask_3 = mask[210,:]
    base = mask[250,:]

    line_1 = np.divide(np.max(np.nonzero(mask_1)) - np.min(np.nonzero(mask_1)), 2)
    line_1 = np.min(np.nonzero(mask_1)) + line_1
    line_2 = np.divide(np.max(np.nonzero(mask_2)) - np.min(np.nonzero(mask_2)), 2)
    line_2 = np.min(np.nonzero(mask_2)) + line_2
    line_3 = np.divide(np.max(np.nonzero(mask_3)) - np.min(np.nonzero(mask_3)), 2)
    line_3 = np.min(np.nonzero(mask_3)) + line_3

    print(line_1, line_2, line_3)

    #cv2.line(img, (320, 30), (320, 30), (255, 255, 255), thickness=5)
    #cv2.line(img, (320, 110), (320, 110), (255, 255, 255), thickness=5)
    #cv2.line(img, (320, 210), (320, 210), (255, 255, 255), thickness=5)

    cv2.line(mask, (line_1, 30),  (line_1, 30),  (0, 255, 255), thickness=5)
    cv2.line(mask, (line_2, 110), (line_2, 110), (0, 255, 255), thickness=5)
    cv2.line(mask, (line_3, 210), (line_3, 210), (0, 255, 255), thickness=5)

    # Central points
    cv2.line(img, (line_center, 30), (line_center, 30), (255, 255, 255), thickness=5)

    return img, line_center, wall




def execute(self):
    
    img = self.getImage()
    # Process image
    img_proc, line_center, wall = processed_image(img)
    error_line = np.subtract(mid, line_center).item()

    global last_error
    global vel_max
    global last_vel
    
    # PID para giro
    giro = kp * error_line + kd * (error_line - last_error)
    self.motors.sendW(giro)

    # PID velocidad
    vel_error = kpv * abs(error_line) + abs(kdv * (error_line - last_error))
    
    # Control de velocidad
    if abs(error_line) in range(0, 15) and wall >= 178:
        cv2.putText(img_proc, str("Line"), (10, 140), font, 0.4, (0,255,0), 2)
        
        cv2.putText(img_proc, str(vel_max), (45, 100), font, 0.4, (0,255,0), 1)
        self.motors.sendV(vel_max)
    elif wall in range(0,179):
        cv2.putText(img_proc, str("Curve"), (10, 140), font, 0.4, (255,255,0), 2)
        if wall < 50:
            brake = 10
        else:
            brake = 5
        vel_correccion = abs(vel_max - vel_error - brake)
        self.motors.sendV(vel_correccion)
        cv2.putText(img_proc, str(vel_correccion), (45, 100), font, 0.4, (255,255,0), 1)
    elif wall == 0:
        cv2.putText(img_proc, str("¿?¿?¿?"), (10, 140), font, 0.4, (255,255,0), 2)
        
        vel_correccion = abs(vel_max - vel_error - (2 * brake))
        self.motors.sendV(vel_correccion)
        cv2.putText(img_proc, str(vel_correccion), (45, 100), font, 0.4, (255,255,0), 1)
    else:
        pass
    
    # Telemetria de giro
    if last_error > error_line: # El error desciende
        cv2.putText(img_proc, str(giro), (45, 120), font, 0.4, (255,0,0), 1)
    elif last_error < error_line: # El error aumenta
        cv2.putText(img_proc, str(giro), (45, 120), font, 0.4, (0,255,0), 1)
    else:
        pass

    last_error = error_line
    last_vel = vel_max

    
    self.set_threshold_image(img_proc)