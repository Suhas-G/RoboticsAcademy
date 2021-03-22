from GUI import GUI
from HAL import HAL
import numpy as np
import cv2
import math
import time



class PID:
    def __init__(self, set_point, kp = 0, ki = 0, kd = 0, offset = 0):
        self.set_point = set_point
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.offset = offset
        self.time = time.time()
        self.error = 0

        self.proportional_part = 0
        self.integral_part = 0
        self.differential_part = 0

    def next(self, pv, time):
        error = self.set_point - pv
        delta_time = time - self.time
        self.proportional_part = self.kp * error
        self.integral_part += (error * delta_time)
        self.integral_part += error
        self.differential_part = self.kd * ((error - self.error) / delta_time)
        self.differential_part = self.kd * (error - self.error)

        self.time = time
        self.error = error
        return self.offset + self.proportional_part + (self.ki * self.integral_part) + self.differential_part

lower_red = np.array([0,50,50])
upper_red = np.array([10,255,255])
kp = 1.7
ki = 8e-6
kd = 7e-7
WIDTH = None
kernel = np.ones((3, 3), dtype=np.uint8)

linear = 2.2
angular = 0


while True:
    # Enter iterative code!
    image = HAL.getImage()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_red, upper_red)
    erosion = cv2.erode(mask, kernel, iterations = 2)
    dilation = cv2.dilate(erosion, kernel, iterations = 2)

    _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if (len(contours) > 0) :
        contour = max(contours, key=cv2.contourArea)
        topmost = tuple(contour[contour[:, :, 1].argmin()][0])
        cv2.circle(image, topmost, 5, (255, 0, 0), -1)

        if WIDTH is None:
            WIDTH = image.shape[1]
            controller = PID(WIDTH // 2, kp=kp, ki=ki, kd=kd)
        else:
            vx = controller.next(topmost[0], time.time()) / WIDTH
            angular = vx * (math.pi/2)
            console.print(topmost)
            HAL.motors.sendV(linear)
            HAL.motors.sendW(angular)
    else:
        HAL.motors.sendV(linear)
        HAL.motors.sendW(angular)

    GUI.showImage(image)