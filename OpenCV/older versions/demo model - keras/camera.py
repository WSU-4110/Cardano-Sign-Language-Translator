# test program of what our project will look like in the end, here just showing the implementation of opencv and
# mediapipe hand_det library with a tensorlfow gesture prediction model

import cv2 
import time
import numpy as np
#import mediapipe as mp
#import tensorflow as tf
#from savefile import *
#from tensorflow.keras.models import load_model
#from AI_model import Net
import torch
from model import Net
#model = torch.load('CNNmodel_version_1.onnx')
#model.eval()

global capture, outputFrame, lock, full_sentence, text_suggestion

signs = {'0': 'A', '1': 'B', '2': 'C', '3': 'D', '4': 'E', '5': 'F', '6': 'G', '7': 'H', '8': 'I',
        '10': 'K', '11': 'L', '12': 'M', '13': 'N', '14': 'O', '15': 'P', '16': 'Q', '17': 'R',
        '18': 'S', '19': 'T', '20': 'U', '21': 'V', '22': 'W', '23': 'X', '24': 'Y' }


#filename = easygui.enterbox("Save file as (press S to start rec):") #save video file

frames_per_second = 24.0
res = '480p'

filename = "yolo"

#Load the gesture recognizer model
model = torch.load('model_trained.pt')
model.eval()
#Load class name of recogniton model
#f = open('gesture.names', 'r')
#classNames = f.read().split('\n')
#f.close()
#print(classNames)   

#width = 1280 
#height = 720

#def res(width, height): #live video
 #   capture.set(3, width);
  #  capture.set(4,height);

capture = cv2.VideoCapture(0)

#out = cv2.VideoWriter(filename, get_type(filename),frames_per_second, get_res(capture, res))

prev_frame = 0

#mpHands = mp.solutions.hands #intialize mediapipe model
#detect_hands = mpHands.Hands(static_image_mode = False, max_num_hands=1, min_detection_confidence = 0.5, min_tracking_confidence = 0.5)
#display = mp.solutions.drawing_utils

cv2.waitKey(0)

while(True):
     
    # Capture the video frame by frame - true or false

    ret, frame = capture.read()

    # Lugar de la imagen donde se toma la muestra
    img = frame[20:250, 20:250]

    res = cv2.resize(img, dsize=(28, 28), interpolation = cv2.INTER_CUBIC)
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    res1 = np.reshape(res, (1, 1, 28, 28)) / 255
    res1 = torch.from_numpy(res1)
    res1 = res1.type(torch.FloatTensor)

    out = model(res1)
    probs, label = torch.topk(out, 25)
    probs = torch.nn.functional.softmax(probs, 1)

    pred = out.max(1, keepdim=True)[1]

    if float(probs[0,0]) < 0.4:
            detected = 'Nothing detected'
    else:
            detected = signs[str(int(pred))] + ': ' + '{:.2f}'.format(float(probs[0,0])) + '%'

    font = cv2.FONT_HERSHEY_SIMPLEX
    frame = cv2.putText(frame, detected, (60,285), font, 1, (255,0,0), 2, cv2.LINE_AA)

    frame = cv2.rectangle(frame, (100, 100), (250, 250), (0, 255, 0), 3)

    # Display the resulting frame
    cv2.imshow('detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('F'):
        break

    while(cv2.waitKey(1) & 0xFF == ord('S')):
        while(cv2.waitKey(1) & 0xFF != ord('s')): #needs fixing only getting screenshot isntead of video
                out.write(frame)
                cv2.imshow('recording',frame)
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
            
#release camera
capture.release()
#delete windows
cv2.destroyAllWindows()


