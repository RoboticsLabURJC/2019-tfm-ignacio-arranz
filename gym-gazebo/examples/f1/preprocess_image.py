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
    
    img = img[220:]
    # Convert to HSV
    img_proc = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #wall_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Get the image processed
    mask = cv2.inRange(img_proc, (1, 235, 60), (180, 255, 255))
    # Get 3 lines from the image.
    wall = img[12][320][0]
    line = mask[30,:]
    base = mask[250,:]
        
    try:
        # Se calcula el centro de la línea roja en la imagen segmentada (valores a 255,255,255)
        line_center = np.divide(np.max(np.nonzero(line)) - np.min(np.nonzero(line)), 2)
        #base_center = np.divide(np.max(np.nonzero(base)) - np.min(np.nonzero(base)), 2)

        line_center = np.min(np.nonzero(line)) + line_center
        #base_center = np.min(np.nonzero(base)) + base_center
    except ValueError:
        line_center = last_center_line
        #base_center = None
        
    # Puntos centrales de la línea segmentada
    cv2.line(img, (line_center, 30), (line_center, 30), (255, 255, 255), thickness=5)
     
    # Puntos centrales de la imagen (verde)
    cv2.line(img, (320, 30),  (320, 30),  (0, 255, 0), thickness=5)
    cv2.line(img, (320, 12),  (320, 12),  (255, 255, 0), thickness=5)
    #cv2.line(img, (320, 250),  (320, 250),  (255, 255, 0), thickness=5)
    
    # Linea diferencia entre punto central - error (blanco)
    cv2.line(img, (320, 30), (line_center, 30),  (255, 0, 0), thickness=2)
    
    # Telemetry
    cv2.putText(img, str("wall: {}".format(wall)), (18, 60), font, 0.4, (255,255,255), 1)
    cv2.putText(img, str("err: {}".format(320 - line_center)), (18, 80), font, 0.4, (255,255,255), 1)
    cv2.putText(img, str("vel: "), (18, 100), font, 0.4, (255,255,255), 1)
    cv2.putText(img, str("turn: "), (10, 120), font, 0.4, (255,255,255), 1)

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