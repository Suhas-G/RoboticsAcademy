from GUI import GUI
from HAL import HAL
import cv2
import numpy as np
# Enter sequential code!

lower_blue = np.array([90, 50, 50], dtype=np.uint8)
upper_blue = np.array([135, 255, 255], dtype=np.uint8)

while True:
    # Enter iterative code!
    frame = HAL.getImage()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    kernel = np.ones((3, 3), dtype=np.uint8)
    erosion = cv2.erode(mask, kernel, iterations = 2)
    dilation = cv2.dilate(erosion, kernel, iterations = 2)

    _, contours, heirarchy = cv2.findContours(dilation, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour_areas = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            contour_areas.append((cnt, area))
    
    sorted_contours = sorted(contour_areas, key=lambda x: x[1], reverse=True)

    for cnt, area in sorted_contours[:2]:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(frame,[box],0,(0,0,255),2)  
        
    GUI.showImage(frame)