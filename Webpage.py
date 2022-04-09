import streamlit as st
import cv2
from mtranslate import translate
import pandas as pd
import os
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
elif choice == "Login":
    username = st.sidebar.text_input("UserName")
    password = st.sidebar.text_input("Password", type='password')
    if st.sidebar.button("Login"):
        create_usertable()
        result = login_user(username,password)
        if result:
            st.success("Logged in as {}".format(username))
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
            langlist = tuple(lang)
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





