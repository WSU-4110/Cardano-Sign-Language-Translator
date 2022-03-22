import numpy as np
import cv2
import torch
from model import Net
import time

cap = cv2.VideoCapture(0)

cap.set(3, 700)
cap.set(4, 480)

model = torch.load('iteration1.pt')
model.eval()

current_frame = 0
prev_frame = 0

signs = {'0': 'A', '1': 'B', '2': 'C', '3': 'D', '4': 'E', '5': 'F', '6': 'G', '7': 'H', '8': 'I',
        '10': 'K', '11': 'L', '12': 'M', '13': 'N', '14': 'O', '15': 'P', '16': 'Q', '17': 'R',
        '18': 'S', '19': 'T', '20': 'U', '21': 'V', '22': 'W', '23': 'X', '24': 'Y' }

while True:
    ret, frame = cap.read()


    current_frame = time.time() #time to finish process of current frame
    FPS = 1/(current_frame - prev_frame)
    prev_frame = current_frame

    FPS = int(FPS)
    FPS = str(FPS) #convert to string for display

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
        output = 'Sign not detected'
    else:
        output = signs[str(int(pred))] + ': ' + '{:.2f}'.format(float(probs[0,0])) + '%'

    font = cv2.FONT_HERSHEY_SIMPLEX
    frame = cv2.putText(frame, output, (60,285), font, 1, (255,0,0), 2, cv2.LINE_AA)
    cv2.putText(frame, FPS, (500,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) #display FPS

    frame = cv2.rectangle(frame, (20, 20), (250, 250), (0, 255, 0), 3)

    cv2.imshow('Cam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
