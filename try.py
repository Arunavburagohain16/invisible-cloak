import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

cap = cv2.VideoCapture(0)
count = 0
while(True):
    #video read part
    ret, frame = cap.read()
    cv2.imshow('frame',frame)
