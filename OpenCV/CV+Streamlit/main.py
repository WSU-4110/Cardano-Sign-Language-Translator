import cv2
import numpy as np
import torch
from model import Net
import time

s = 0 # Use webcam camera
video_cap = cv2.VideoCapture(s)

video_cap.set(3, 700)
video_cap.set(4, 480)

win_name = 'Camera Preview'
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
  
model = torch.load('iteration1.pt')
model.eval()

current_frame = 0
prev_frame = 0

signs = {'0': 'A', '1': 'B', '2': 'C', '3': 'D', '4': 'E', '5': 'F', '6': 'G', '7': 'H', '8': 'I',
            '10': 'K', '11': 'L', '12': 'M', '13': 'N', '14': 'O', '15': 'P', '16': 'Q', '17': 'R',
            '18': 'S', '19': 'T', '20': 'U', '21': 'V', '22': 'W', '23': 'X', '24': 'Y' }


# Create a network object
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

# Model parameters used to train model.
mean = [104, 117, 123]
scale = 1.0
in_width = 300
in_height = 300

# Set the detection threshold for face detections.
detection_threshold = 0.8

# Annotation settings.
font_style = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
font_thickness = 1

while True:
    has_frame, frame = video_cap.read()

    width = 700
    height = 480
            
    h = frame.shape[0]
    w = frame.shape[1]
    # Flip THE video frame horizontally (not required, just for convenience)
    frame = cv2.flip(frame, 1)

    current_frame = time.time() #time to finish process of current frame
    FPS = 1/(current_frame - prev_frame)
    prev_frame = current_frame

    FPS = int(FPS)
    FPS = str(FPS) #convert to string for display


    # Convert the image into a blob
    blob = cv2.dnn.blobFromImage(frame, scalefactor=scale, size=(in_width, in_height), mean=mean, swapRB=False, crop=False)
    # Pass the blob to the DNN model.
    net.setInput(blob)
    # Retrieve detections from the DNN model.
    detections = net.forward()

    # Process each detection.
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > detection_threshold:
            # Extract the bounding box coordinates from the detection.
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype('int')
            a1 = x1 - 250
            b1 = y1 - 20
            a2 = x2 - 250
            b2 = y2 + 20

            frame = cv2.resize( frame, (w,h))

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
            # Annotate the video frame with the detection results.
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.rectangle(frame, (20,20), (250, 250), (0, 255, 0), 3)


            # roi = frame[b1:b2, a1:a2]
            # onnx_model_path ="modelResNet5.onnx"
            # net = cv2.dnn.readNetFromONNX(onnx_model_path)
            # print(net)
            # blob = cv2.dnn.blobFromImage(roi, scalefactor=1, size=(200,200), mean=(0,0,0), swapRB=True, crop=True)
            # net.setInput(blob)
            # preds = net.forward()
            #
            # index_to_letter = {'0': 'A', '1': 'B', '2': 'C', '3': 'D', '4': 'E', '5': 'F', '6': 'G', '7': 'H', '8': 'I',
            #                    '10': 'K', '11': 'L', '12': 'M', '13': 'N', '14': 'O', '15': 'P', '16': 'Q', '17': 'R',
            #                    '18': 'S', '19': 'T', '20': 'U', '21': 'V', '22': 'W', '23': 'X', '24': 'Y'}
            #
            # biggest_pred_index = np.array(preds)[0].argmax()
            # print("Predicted class:", index_to_letter[str(int(biggest_pred_index))])
            # print("Predicted class:", str(int(biggest_pred_index)))
            #
            # label = index_to_letter[str(int(biggest_pred_index))]
            # label_size, base_line = cv2.getTextSize(label, font_style, font_scale, font_thickness)
            # cv2.rectangle(frame, (x1, y1 - label_size[1]), (x1 + label_size[0], y1 + base_line), (255, 255, 255),
            #               cv2.FILLED)
            # cv2.putText(frame, label, (x1, y1), font_style, font_scale, (0, 0, 0))

    cv2.imshow(win_name, frame)
    key = cv2.waitKey(1)
    if key == ord('Q') or key == ord('q') or key == 27:
        break
video_cap.release()
cv2.destroyWindow(win_name)

