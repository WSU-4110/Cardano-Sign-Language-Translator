#test program of what our project will look like in the end, here just showing the implementation of opencv and mediapipe hand_det library

import cv2 
import time
import numpy as np
import mediapipe as mp


width = 720 
height = 1440

def res(width, height): #live video
    capture.set(3, width);
    capture.set(4,height);

capture = cv2.VideoCapture(0)

prev_frame = 0

mpHands = mp.solutions.hands
detect_hands = mpHands.Hands(static_image_mode = False, max_num_hands=2, min_detection_confidence = 0.5, min_tracking_confidence = 0.5)
display = mp.solutions.drawing_utils


cv2.waitKey(0)

while(True):
     
    # Capture the video frame by frame - true or false
    ret, frame = capture.read()
    frame.flags.writeable = False
    RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #convert frame to RGB

    result = detect_hands.process(RGB)

    current_frame = time.time() #time to finish process of current frame
    FPS = 1/(current_frame - prev_frame)
    prev_frame = current_frame

    FPS = int(FPS)
    FPS = str(FPS) #convert to string for display

    if result.multi_hand_landmarks: #connect where key points are detected
        for hands_l in result.multi_hand_landmarks:
            for id, lm in enumerate(hands_l.landmark): #every point in hand has associated id 
                #print(id,lm) #print cordinates in termianal, slows performance
                h, w, c = frame.shape #height, width, coordinate of point
                cx, cy = int(lm.x *w), int(lm.y*h) #attempting to find key points

            display.draw_landmarks(frame, hands_l, mpHands.HAND_CONNECTIONS)#display on screen

    cv2.putText(frame, FPS, (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) #display FPS
    
    # Display the resulting frame
    cv2.imshow('frame', frame)
      
    if cv2.waitKey(1) & 0xFF == ord('F'):
        break
  
#release camera
capture.release()
#delete windows
cv2.destroyAllWindows()