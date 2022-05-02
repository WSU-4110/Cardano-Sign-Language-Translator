import cv2 
import time
import numpy as np
#from savefile import *
from model import Net
import torch
import onnx
from onnx2pytorch import ConvertModel
from torchvision import transforms  


cap = cv2.VideoCapture(0)

cap.set(3, 700)
cap.set(4, 480)


onnx_model = onnx.load("CNNmodel_version_1.onnx")
onnx.checker.check_model(onnx_model)


pytorch_model = ConvertModel(onnx_model)

signs = {'0': 'A', '1': 'B', '2': 'C', '3': 'D', '4': 'E', '5': 'F', '6': 'G', '7': 'H', '8': 'I',
        '10': 'K', '11': 'L', '12': 'M', '13': 'N', '14': 'O', '15': 'P', '16': 'Q', '17': 'R',
        '18': 'S', '19': 'T', '20': 'U', '21': 'V', '22': 'W', '23': 'X', '24': 'Y' }


#filename = easygui.enterbox("Save file as (press S to start rec):") #save video file


current_frame = 0
prev_frame = 0

while True:

    current_frame = time.time() #time to finish process of current frame
    FPS = 1/(current_frame - prev_frame)
    prev_frame = current_frame

    FPS = int(FPS)
    FPS = str(FPS) #convert to string for display

    ret, frame = cap.read()


    #put video in certain frame part
    img = frame[20:250, 20:250]
    frame = cv2.flip(frame, 1)

    

    #res = cv2.resize(img, dsize=(28, 28), interpolation = cv2.INTER_CUBIC) #changing resolution
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #turning RGB

    res1 = np.reshape(frame, (28, 1, 3, 3)) / 255 #change array to res1 without changing conten  t
    res1 = transforms.ToTensor()(np.array(res1))
    res1 = res.type(torch.FloatTensor) #tensor to double type
    out = pytorch_model(res1)

    # probability
    probs, label = torch.topk(out, 25) #return largest element with dimension
    probs = torch.nn.functional.softmax(probs, 1) #make sure all element lie in range [0-1] and sum to 1 as a max

    pred = out.max(1, keepdim=True)[1] #reuturn the max value of all elements in input tensor in given dimension

    if float(probs[0,0]) < 0.4:
        text = 'Sign not detected'
    else:
        text = signs[str(int(pred))] + ': ' + '{:.2f}'.format(float(probs[0,0])) + '%'

    font = cv2.FONT_HERSHEY_SIMPLEX
    frame = cv2.putText(frame, text, (60,285), font, 1, (255,0,0), 2, cv2.LINE_AA)
    cv2.putText(frame, FPS, (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) #display FPS

    frame = cv2.rectangle(frame, (20, 20), (250, 250), (0, 255, 0), 3)
    
    cv2.imshow('Cam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
