import streamlit as st
import cv2
from mtranslate import translate #pip install mtranslate
import pandas as pd
import os


# --- Header section -- #
st.set_page_config(layout="wide")
st.title("Hi, Welcome to Cardano!")
# setup camera on streamlit
st.subheader("Camera Feed")
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)
while run:
    _, frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)
else:
    st.write('Stop')

# pip install openpyxl
df = pd.read_excel(os.path.join('language.xlsx'), sheet_name='wiki')
df.dropna(inplace=True)
lang = df['name'].to_list()
langlist=tuple(lang)
langcode = df['iso'].to_list()

lang_array = {lang[i]: langcode[i] for i in range(len(langcode))}
input_text = st.text_area("Sign Language Translation", height=200)
option = st.sidebar.radio('SELECT LANGUAGE', langlist)
if len(input_text) > 0:
    try:
        output = translate(input_text, lang_array[option])
        st.text_area("Translated Text", output, height=200)
    except Exception as e:
        st.error(e)


