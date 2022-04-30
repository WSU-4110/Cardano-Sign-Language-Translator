import cv2
import numpy as np
import torch
from AI import*
import time
import autocomplete
import sys
from google_trans_new import google_translator  
import threading
import argparse
import imutils
import streamlit as st
import unittest



#def show_device(self):
#   return str(self.device)

# Method moves the model or tensor to either gpu or cpu for computing
# Both the model and the tensor need to be in the same device


def translation():
    sentence = "hello"
    convert = "de"
    return "hallo "



# Method to predict the image
def predict_image():
    #device = torch.device('cpu')
    # Convert to a batch of 1
    #xb = to_device(img.unsqueeze(0), device)
    # Get predictions from model
    #yb = model(xb)
    # Pick index with highest probability
    #_, preds  = torch.max(yb, dim=1)
    # Retrieve the class label
    return 'A'


def get_suggestion():
    #autocomplete.load()
    #full_sentence = "my name is"
    #separated = full_sentence.strip().split(' ')

    #print(separated)

    #if(len(separated)==0):
        #return ['i', 'me', 'the', 'my', 'there']
    #elif(len(separated)>=2):
      #  first = ''
       # second = ''

        #first = separated[-2]
        #second = separated[-1]
        
        #suggestions = autocomplete.predict(first, second)[:5]
        
    return "my name"




def detection():
    full_sentence = ''
    text_suggestion = ''
    letter =''
    language =''
    current_frame = 0
    prev_frame = 0
    win_name = "Detection"
            
    #model = ResNet152()
    # Load pretrained model
    #model.load_state_dict(torch.load('modelResNet11.pth', map_location=torch.device('cpu')))
    # Set Model to inferecning mode
    #model.eval()

    #autocomplete.load()


    #device = torch.device('cpu')


    index_to_letter={ 0:'A', 1:'B', 2:'C', 3:'D', 4: 'D', 5:'E',
                     6:'G', 7:'H', 8:'I', 9:'J', 10:'K', 11:'K',
                     12:'L', 13:'N', 14:'O', 15:'P', 16:'Q', 17:'R',
                     18:'S', 19:'T', 20:'U', 21:'V', 22:'W', 23:'X',
                     24: 'Y', 25:'Z', 26:'del', 27:'nothing', 28:'space'}


    #device = torch.device('cpu')
    #video_cap = cv2.VideoCapture(0)

    #video_cap.set(3, 700)
    #video_cap.set(4, 480)


    while True:
        key = cv2.waitKey(2)
     

        width = 700
        height = 480
                
        
        # Flip THE video frame horizontally (not required, just for convenience)

        #current_frame = time.time() #time to finish process of current frame
        #FPS = 1/(current_frame - prev_frame)
        #prev_frame = current_frame

        #FPS = int(FPS)
        #FPS = str(FPS) #convert to string for display

        #frame = cv2.resize( frame, (width,height))

        #frame = img[20:250, 20:250]

        #img = Image.fromarray(img)

        #img = tt.functional.rotate(img, angle=0)

        
        #preprocessing
        #transform_img = tt.Compose([tt.Resize((200, 200)),
                         #                   tt.ToTensor(),
                          #                  tt.Normalize([0.485,0.456,0.406],[0.299,0.224,0.224])])
        #img = transform_img(img)


        #xb = to_device(img.unsqueeze(0), device)
        # Get predictions from model
        #yb = model(xb)

        #probs, label = torch.topk(yb, 25)
        #probs = torch.nn.functional.softmax(probs, 1)

      
        letter = 'A'

        if(letter == 'space' or letter == 'nothing' or letter == 'del'):
            output = "Nothing detected"
        #output = letter + ': ' + '{:.2f}'.format(float(probs[0,0])*100) + '%'
        else:
            output = letter


      
                          
        #font = cv2.FONT_HERSHEY_SIMPLEX
        #frame = cv2.putText(frame, output, (60,285), font, 1, (255,0,0), 2, cv2.LINE_AA)
        #cv2.rectangle(frame, (20,20), (250, 250), (0, 255, 0), 3)
        #cv2.namedWindow(win_name)        # Create a named window

        #cv2.moveWindow(win_name, 1000,200)
        
        #cv2.imshow(win_name, frame)            
        #if key == ord('Q') or key == ord('q') or key == 27:
        #cv2.destroyAllWindows()
        break
    return True

        
    



    








