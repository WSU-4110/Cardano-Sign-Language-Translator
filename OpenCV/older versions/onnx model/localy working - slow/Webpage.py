import streamlit as st
import cv2
import pandas as pd
import os
import numpy as np
import torch
import threading
from model import Net
import time
import autocomplete
from google_trans_new import google_translator  
from gtts import gTTS


def get_suggestion(prev_word='my', next_semi_word='na'):
    global full_sentence
    separated = full_sentence.strip().split(' ')

    print(separated)

    if(len(separated)==0):
        return ['i', 'me', 'the', 'my', 'there']
    elif(len(separated)==1):
        suggestions = autocomplete.predict(full_sentence, '')[:5]
    elif(len(separated)>=2):
        first = ''
        second = ''

        first = separated[-2]
        second = separated[-1]
        
        suggestions = autocomplete.predict(first, second)[:5]
        
    return [word[0] for word in suggestions]


model = torch.load('iteration1.pt')
model.eval()

current_frame = 0
prev_frame = 0

full_sentence = ''
text_suggestion = ''

signs = {'0': 'A', '1': 'B', '2': 'C', '3': 'D', '4': 'E', '5': 'F', '6': 'G', '7': 'H', '8': 'I',
            '10': 'K', '11': 'L', '12': 'M', '13': 'N', '14': 'O', '15': 'P', '16': 'Q', '17': 'R',
            '18': 'S', '19': 'T', '20': 'U', '21': 'V', '22': 'W', '23': 'X', '24': 'Y' }

autocomplete.load() #text suggestion in future

lock = threading.Lock()


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
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
font_thickness = 1


    
# --- Header section -- #
st.set_page_config(layout="wide")
st.title("Hi, Welcome to Cardano!")
# setup camera on streamlit
st.subheader("Camera Feed")
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)
camera.set(3, 700)
camera.set(4, 480)



while run:
    has_frame, frame = camera.read()  
        
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
            b1 = y1 - 30
            a2 = x2 - 250
            b2 = y2 + 30

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

            key = cv.waitKey(0);
 
            if key == ord('D') or key == ord('d') or key = 27:
                full_sentence+=signs[str(int(pred))].lower()

            if(text_suggestion!=''):
                if(text_suggestion==' '):
                    full_sentence+=' '
                    text_suggestion=''
                else:
                    full_sentence_list = full_sentence.strip().split()
                    if(len(full_sentence_list)!=0):
                        full_sentence_list.pop()
                    full_sentence_list.append(text_suggestion)
                    full_sentence = ' '.join(full_sentence_list)
                    full_sentence+=' '
                    text_suggestion=''

            frame = cv2.putText(frame, output, (60,285), font, 1, (255,0,0), 2, cv2.LINE_AA)
            cv2.putText(frame, FPS, (500,70), font, 1, (0, 255, 0), 2) #display FPS
            # Annotate the video frame with the detection results.
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.rectangle(frame, (20,20), (250, 250), (0, 255, 0), 3)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  


    FRAME_WINDOW.image(frame)
    
else:
    st.write('Stop')

st.subheader("Suggestion Feed")
st.write(f"This will be our suggestion feed")


#integrate Google Cloud API

trans = google_translator() 

# pip install openpyxl
df = pd.read_excel(os.path.join('language.xlsx'), sheet_name='wiki')
df.dropna(inplace=True)
lang = df['name'].to_list()
langlist=tuple(lang)
langcode = df['iso'].to_list()


text = st.text_input('Sign Language Translation')
#option = st.sidebar.radio('SELECT LANGUAGE', langlist)

language = st.sidebar.selectbox('SELECT LANGUAGE', langlist)


langcode = "";

for row in df.rows:
    if(df[row][0] == language):
        langcode+=df[row][1]




if st.button('Translate'):    
        result = trans.translate(text, lang_tgt= langcode)
        st.write(result)
        speech = gTTS(text = result, lang = langcode, slow = False)
        speech.save('user_trans.mp3')          
        audio_file = open('user_trans.mp3', 'rb')    
        audio_bytes = audio_file.read()    
        st.audio(audio_bytes, format='audio/ogg',start_time=0)


