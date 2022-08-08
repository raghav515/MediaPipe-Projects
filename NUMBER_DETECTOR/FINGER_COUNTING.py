import cv2
import time
import os
wCam,hCam=648,488
cap=cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)

myList=os.listdir()

while True:
    success, img=cap.read()
    cv2.imshow("Image",img)
    cv2.waitKey(1)