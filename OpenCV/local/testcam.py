import cv2
import numpy as np
import torch
from AI import*
import time
import autocomplete
import sys
from google_trans_new import google_translator  




    
    





# Method moves the model or tensor to either gpu or cpu for computing
# Both the model and the tensor need to be in the same device
def to_device(data, device):
    # Move Tensors to a chosen device
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

# Method to predict the image
def predict_image(img, model):
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    _, preds  = torch.max(yb, dim=1)
    # Retrieve the class label
    return preds[0]


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

# Instantiate Model
model = ResNet152()
# Load pretrained model
model.load_state_dict(torch.load('modelResNet10.pth', map_location=torch.device('cpu')))
# Set Model to inferecning mode
model.eval()

autocomplete.load()

# Get the device your machine has, either cuda(gpu) or cpu
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


index_to_letter={ 0:'A', 1:'B', 2:'C', 3:'D', 4: 'D', 5:'E',
                 6:'G', 7:'H', 8:'I', 9:'J', 10:'K', 11:'K',
                 12:'L', 13:'N', 14:'O', 15:'P', 16:'Q', 17:'R',
                 18:'S', 19:'T', 20:'U', 21:'V', 22:'W', 23:'X',
                 24: 'Y', 25:'Z', 26:'del', 27:'nothing', 28:'space'}



s = 0 # Use webcam camera
video_cap = cv2.VideoCapture(s)

video_cap.set(3, 700)
video_cap.set(4, 480)


current_frame = 0
prev_frame = 0

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


full_sentence = ''
text_suggestion = ''


while True:
    key = cv2.waitKey(2)
    has_frame, frame = video_cap.read()

    width = 700
    height = 480
            
    h = frame.shape[0]
    w = frame.shape[1]
    # Flip THE video frame horizontally (not required, just for convenience)

    current_frame = time.time() #time to finish process of current frame
    FPS = 1/(current_frame - prev_frame)
    prev_frame = current_frame

    FPS = int(FPS)
    FPS = str(FPS) #convert to string for display

    frame = cv2.resize( frame, (width,height))

    img = frame[20:250, 20:250]

    img = Image.fromarray(img)

    img = tt.functional.rotate(img, angle=0)

    
    #preprocessing
    transform_img = tt.Compose([tt.Resize((225, 225)),
                                        tt.RandomHorizontalFlip(p=0.3), 
                                        tt.RandomRotation(30),
                                        tt.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                                        tt.RandomPerspective(distortion_scale=0.2),
                                        tt.ToTensor(),
                                        tt.Normalize([0.485, 0.456, 0.406],
                                        [0.229, 0.224, 0.225])])
    img = transform_img(img)


    xb = to_device(img.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)

    probs, label = torch.topk(yb, 25)
    probs = torch.nn.functional.softmax(probs, 1)

    if float(probs[0][0]) < 0.4:
        output = "Nothing detected"

    else:
        letter_index = int(str(predict_image(img, model).numpy()))
        letter = index_to_letter.get(letter_index)

        #if(letter == "space" || letter == "nothing" || letter == "del"):
            #output = "Nothing detected"
        #output = letter + ': ' + '{:.2f}'.format(float(probs[0,0])*100) + '%'
        output = letter


    option = 0     

    if key == 49:
            option = 1
            print(option)

    if key == 50:
            option = 2
            print(option)


    if key == 51:
            option = 3
            print(option)


    if key == 52:
            option = 4
            print(option)


    if key == 53:
            option = 5
            print(option)


        
    if key == ord('E') or key == ord('e') or key == 27:

        full_sentence+=letter.lower()
        
        recommended = get_suggestion()

        print(recommended)

        print(len(recommended))

    if(option > 0):
        text_suggestion=recommended[option-1]
        print(text_suggestion)

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
                      
    font = cv2.FONT_HERSHEY_SIMPLEX
    frame = cv2.putText(frame, output, (60,285), font, 1, (255,0,0), 2, cv2.LINE_AA)
    cv2.rectangle(frame, (20,20), (250, 250), (0, 255, 0), 3)

    cv2.imshow("read", frame)            


    if key == ord('Q') or key == ord('q') or key == 27:
        break


sentence = ""

for x in range (len(full_sentence)):
    sentence+=full_sentence[x]
print(sentence)

trans = google_translator() 

# pip install openpyxl
df = pd.read_excel(os.path.join('language.xlsx'), sheet_name='wiki')
df.dropna(inplace=True)
lang = df['name'].to_list()
langlist=tuple(lang)
langcode = df['iso'].to_list()

result = trans.translate(sentence, lang_tgt= 'hr')
print(result)
        

video_cap.release()
#cv2.destroyWindow(win_name)

