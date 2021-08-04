import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

cap = cv2.VideoCapture(0)
count = 0
while(True):
    #video read part
    ret, frame = cap.read()
    frame = np.flip(frame,axis = 1)
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    #Range for lower blue
    blue_lower=np.array([100,150,0],np.uint8)
    blue_upper=np.array([140,255,255],np.uint8)
    mask1 = cv2.inRange(hsv,blue_lower,blue_upper)
    if(count < 100):
        cv2.imwrite('frame_back.jpg',frame)
    background = cv2.imread('frame_back.jpg')
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, np.ones((3,3),np.uint8))
    #creating an inverted mask to segment out the cloth from the frame
    mask2 = cv2.bitwise_not(mask1)
    #Segmenting the cloth out of the frame using bitwise and with the inverted mask
    res1 = cv2.bitwise_and(frame,frame,mask=mask2)
    # creating image showing static background frame pixels only for the masked region
    res2 = cv2.bitwise_and(background, background, mask = mask1)
    #Generating the final output
    final_output = cv2.addWeighted(res1,1,res2,1,0)
    cv2.imshow('frame',final_output)
    count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
