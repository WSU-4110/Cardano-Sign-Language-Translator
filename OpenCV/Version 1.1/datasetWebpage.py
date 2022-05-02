from AI import *
import autocomplete
from google_trans_new import google_translator
import streamlit as st
from gtts import gTTS
import sqlite3

# Database
conn = sqlite3.connect('data.db')
c = conn.cursor()
def create_usertable():
    c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT, password)')

def add_userdata(username,password):
    c.execute('INSERT INTO userstable(username,password) VALUES(?,?)',(username, password))
    conn.commit()

def login_user(username,password):
    c.execute('SELECT * FROM userstable WHERE username =? AND password =?', (username, password))
    data = c.fetchall()
    return data

def view_all_users():
    c.execute('SELECT * FROM userstable')
    data = c.fetchall()
    return data

st.set_page_config(layout="wide")
st.title("Hi, Welcome to Cardano!")

# --- Header section -- #
menu = ("Home","Login","Signup")
choice = st.sidebar.selectbox("Menu",menu)
if choice == "Home":
    st.subheader("About Cardano")
    st.write(' In America, less than 1% of the population knows sign language. '
             ' Through Cardano we hoped to remove the communication between the hearing '
             ' and Non-Hearing individuals. By combining a Pytorch Deep learning module ' 
             ' and OpenCV image processing, we are able to create a neural network model for image classification/pose recognition. '
             ' Overall, we hope that through our application deaf or mute people are able to communicate '
             ' more easily.')
    image = Image.open('Images/ex.jpg')
    st.image(image, caption='An example of the application detecting ASL')

elif choice == "Login":
    username = st.sidebar.text_input("UserName")
    password = st.sidebar.text_input("Password", type='password')
    if st.sidebar.checkbox("Login"):
        create_usertable()
        result = login_user(username,password)
        if result:
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
                _, preds = torch.max(yb, dim=1)
                # Retrieve the class label
                return preds[0]


            def get_suggestion(prev_word='my', next_semi_word='na'):
                global full_sentence
                separated = full_sentence.strip().split(' ')

                print(separated)

                if (len(separated) == 0):
                    return ['i', 'me', 'the', 'my', 'there']
                elif (len(separated) == 1):
                    suggestions = autocomplete.predict(full_sentence, '')[:5]
                elif (len(separated) >= 2):
                    first = ''
                    second = ''

                    first = separated[-2]
                    second = separated[-1]

                    suggestions = autocomplete.predict(first, second)[:5]

                return [word[0] for word in suggestions]


            def detection():
                global full_sentence, text_suggestion, letter, language, run, win_name
                while run:
                    key = cv2.waitKey(2)
                    has_frame, frame = video_cap.read()

                    width = 700
                    height = 480

                    # Flip THE video frame horizontally (not required, just for convenience)

                    # current_frame = time.time() #time to finish process of current frame
                    # FPS = 1/(current_frame - prev_frame)
                    # prev_frame = current_frame

                    # FPS = int(FPS)
                    # FPS = str(FPS) #convert to string for display

                    # frame = cv2.resize( frame, (width,height))

                    img = frame[20:250, 20:250]

                    img = Image.fromarray(img)

                    img = tt.functional.rotate(img, angle=0)

                    # preprocessing
                    transform_img = tt.Compose([tt.Resize((200, 200)),
                                                tt.ToTensor(),
                                                tt.Normalize([0.485, 0.456, 0.406], [0.299, 0.224, 0.224])])
                    img = transform_img(img)

                    xb = to_device(img.unsqueeze(0), device)
                    # Get predictions from model
                    yb = model(xb)

                    probs, label = torch.topk(yb, 25)
                    probs = torch.nn.functional.softmax(probs, 1)

                    if float(probs[0][0]) < 0.20:
                        output = "Nothing detected"

                    else:
                        if device == 'cpu':
                            letter_index = int(str(predict_image(img, model).numpy()))
                        else:
                            letter_index = (predict_image(img, model)).item()

                        letter = index_to_letter.get(letter_index)

                        if (letter == 'space' or letter == 'nothing' or letter == 'del'):
                            output = "Nothing detected"
                        # output = letter + ': ' + '{:.2f}'.format(float(probs[0,0])*100) + '%'
                        else:
                            output = letter

                    option = -1

                    if key == 48:
                        option = 0

                    if key == 49:
                        option = 1

                    if key == 50:
                        option = 2

                    if key == 51:
                        option = 3

                    if key == 52:
                        option = 4

                    if key == ord('E') or key == ord('e') or key == 27:
                        full_sentence += output.lower()

                        recommended = get_suggestion()

                        st.write(recommended)  # make list look nicer
                        st.info("Press the corresponding key on keyboard 0-4 to select suggested word")
                        # print(len(recommended))

                    if (option > -1):
                        text_suggestion = recommended[option]
                        # print(text_suggestion)
                        st.warning(text_suggestion)
                        st.info(
                            "Press q to confirm full sentence and begin translation or show sign and press E again to give another word")
                    #
                    # elif(key==ord('R') or key==ord('r')):
                    #     text_suggestion=output
                    #     continue


                    if (text_suggestion != ''):
                        if (text_suggestion == ' '):
                            full_sentence += ' '
                            text_suggestion = ''
                        else:
                            full_sentence_list = full_sentence.strip().split()
                            if (len(full_sentence_list) != 0):
                                full_sentence_list.pop()
                            full_sentence_list.append(text_suggestion)
                            full_sentence = ' '.join(full_sentence_list)
                            full_sentence += ' '
                            text_suggestion = ''

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    frame = cv2.putText(frame, output, (60, 285), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    cv2.rectangle(frame, (20, 20), (250, 250), (0, 255, 0), 3)
                    cv2.namedWindow(win_name)  # Create a named window

                    cv2.moveWindow(win_name, 1000, 200)

                    cv2.imshow(win_name, frame)
                    if key == ord('Q') or key == ord('q') or key == 27:
                        cv2.destroyAllWindows()
                        break


            # Annotation settings.
            font_style = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1

            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            # Instantiate Model
            model = ResNet152()
            # Load pretrained model
            if device == 'cuda':
                model.load_state_dict(torch.load('modelResNet11.pth'))
            else:
                model.load_state_dict(torch.load('modelResNet11.pth', map_location=torch.device('cpu')))

            model = to_device(model, device)
            # Set Model to inferecning mode
            model.eval()

            autocomplete.load()

            index_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'D', 5: 'E',
                               6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'K',
                               12: 'L', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R',
                               18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X',
                               24: 'Y', 25: 'Z', 26: 'del', 27: 'nothing', 28: 'space'}

            full_sentence = ''
            text_suggestion = ''
            letter = ''
            language = ''
            current_frame = 0
            prev_frame = 0
            win_name = "Detection"

            video_cap = cv2.VideoCapture(1) # If you have an external Webcam: Switch to 1 if 0 doesn't work and vice versa. Otherwise, keep it 0

            video_cap.set(3, 700)
            video_cap.set(4, 480)

            # --- Header section -- #
            # st.set_page_config(layout="wide")
            # st.title("Hi, Welcome to Cardano!")
            # setup camera on streamlit

            # pip install openpyxl
            df = pd.read_excel(os.path.join('language.xlsx'), sheet_name='wiki')
            df.dropna(inplace=True)
            lang = df['name'].to_list()
            langlist = tuple(lang)
            langcode = df['iso'].to_list()

            uniqueKey = 1
            run = st.checkbox('Run', key=(uniqueKey+1))

            st.subheader("Select language to translate")
            select = st.selectbox("Select lanugage", langlist, 0)

            convert = df.loc[df[df['name'] == select].index[0], 'iso']

            print(convert)

            if convert != 'Select' and run:
                st.subheader("Suggestion feed")

                st.warning("make sure detection app is selected by mouse")
                st.info("Press E on the keyboard to take desired letter")

                detection()  # runs detection algorithm

            sentence = ""
            if run:
                for x in range(len(full_sentence)):
                    sentence += full_sentence[x]

            if run:
                st.subheader("Full Sentence")
                st.success(sentence)
                trans = google_translator()
                st.subheader("Language selected")

                # st.write("press number for language chosen:\n 1. hr\n 2. fr \n 3. de\n 4. la\n")

                # select = input("input here: ")

                st.write(select)

                # print(select)

                # chosen =""
                # option = int(select)

                # if option == 1:
                #   chosen = 'hr'
                # if option == 2:
                #   chosen = 'fr'
                # if option == 3:
                #   chosen = 'de'
                # if option == 4:
                #   chosen = 'la'

                # st.write("Language chosen: ")
                # st.write(select)

                st.subheader("Translation")

                if (convert != 'Select') and (run):
                    result = trans.translate(sentence, lang_tgt=convert)
                    st.success(result)
                    st.subheader("Audio")
                    speech = gTTS(text=result, lang=convert, slow=False)
                    speech.save('user_trans.mp3')
                    audio_file = open('user_trans.mp3', 'rb')
                    audio_bytes = audio_file.read()
                    st.audio(audio_bytes, format='audio/ogg', start_time=0)

        else:
            st.warning("Incorrect Username/Password")


elif choice == "Signup":
    st.sidebar.subheader("Create New Account")
    new_user = st.sidebar.text_input("Username")
    new_password = st.sidebar.text_input("Password", type='password')

    if st.sidebar.button("Signup"):
        create_usertable()
        add_userdata(new_user, new_password)
        st.success("You have successfully created a Valid Account")
        st.info("Go to Login menu to login")



