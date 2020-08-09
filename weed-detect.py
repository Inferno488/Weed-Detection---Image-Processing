# -*- coding: utf-8 -*-
"""
Created on Thu July 18  21:31:44 2019

@author: Prabhleen
VERSION ALPHA 2
"""

import cv2
import numpy as np
import sys

#FUNCTIONS START---------------------------------------------------------------
imgcopy = 0
def main():
    cv2.namedWindow('ControlPanel',cv2.WINDOW_NORMAL)
    #TRACKBARS FOR DEBUGGING
    cv2.createTrackbar('Hue min', 'ControlPanel', 30, 180, update)
    cv2.createTrackbar('Hue max', 'ControlPanel', 180, 180, update)
    cv2.createTrackbar('Sat min', 'ControlPanel', 60, 255, update)
    cv2.createTrackbar('Sat max', 'ControlPanel', 255, 255, update)
    cv2.createTrackbar('Val min', 'ControlPanel', 0, 255, update)
    cv2.createTrackbar('Val max', 'ControlPanel', 255, 255, update)
    update()

        


def filter(img):
    kernel = np.ones((7,7),np.uint8)
    erode = cv2.erode(img,kernel,iterations = 1)

    dilate = cv2.dilate(erode, kernel,iterations = 2)

    result = cv2.GaussianBlur(dilate,(3,3),1)
    return result


def update(*arg):
    global imgcopy
    #GETTING POSITION OF TRACKBARS
    h0 = cv2.getTrackbarPos('Hue min', 'ControlPanel')
    h1 = cv2.getTrackbarPos('Hue max', 'ControlPanel')
    s0 = cv2.getTrackbarPos('Sat min', 'ControlPanel')
    s1 = cv2.getTrackbarPos('Sat max', 'ControlPanel')
    v0 = cv2.getTrackbarPos('Val min', 'ControlPanel')
    v1 = cv2.getTrackbarPos('Val max', 'ControlPanel')
    lower = np.array((h0,s0,v0))
    upper = np.array((h1,s1,v1))
    
    #Filtering
    mask = cv2.inRange(hsv, lower, upper)
    output = filter(mask)
    
    
    #DETECTING CONTOURS
    contours,hierarchy = cv2.findContours(output,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    
    totalArea = 0
    areas = []
    outputFinal = img.copy()
    imgcopy = outputFinal
    for i in contours:
        area = cv2.contourArea(i)
        totalArea += area
        if (area > 4000 and area < 52500):
            areas.append(i)
        
    for i in areas:
        (x,y),rad = cv2.minEnclosingCircle(i)
        center = (int(x),int(y))
        rad = int(rad)
        cv2.circle(outputFinal,center,rad,(51,255,255),1)
        cv2.circle(outputFinal,center,3,(0,0,255),5)
        cv2.drawContours(outputFinal, i, -1, (255, 255, 255), 1)
        
    #DISPLAYING GUI AND STUFF
    text = "Areas:["+str(len(areas))+"]"
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(outputFinal,text,(10,30), font, .4, (255, 255, 255), 1)
    #cv2.imshow("Original",img)
    #cv2.imshow("Masked",mask)
    #cv2.imshow("Filtered",output)
    #cv2.imshow("Detection",outputFinal)
    
#FUNCTIONS END-----------------------------------------------------------------
#EXECUTION---------------------------------------------------------------------

cap = cv2.VideoCapture(0)
while(True):
    ret,src = cap.read()
    height,width = src.shape[:2]
    if((height > 1000) & (width > 1000)):
        img = cv2.resize(src,None,fx = 0.15,fy= 0.15,interpolation=cv2.INTER_CUBIC)
    else:
        img = src
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    main()
    cv2.imshow("video",imgcopy)
    if (cv2.waitKey(25) & 0xFF == ord('q') ):
        break

cv2.destroyAllWindows()

#EXECUTION---------------------------------------------------------------------
