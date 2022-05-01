import streamlit as st
import cv2
from mtranslate import translate
import pandas as pd
import os
import sqlite3
from PIL import Image
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
    st.subheader("About Cardano...")
    st.write(' In America, less than 1% of the population knows sign language. '
             ' Through Cardano we hoped to remove the communication between the hearing '
             ' and Non-Hearing individuals. By combining a Pytorch Deep learning module ' 
             ' and OpenCV image processing, we are able to create a neural network model for pose recognition. '
             ' Overall, we hope that through our application deaf or mute people are able to communicate '
             ' more easily.')
    image2 = Image.open('ex.jpg')
    image = Image.open('sign.jpg')
    st.image(image2)
    st.image(image)
elif choice == "Login":

    username = st.sidebar.text_input("UserName")
    password = st.sidebar.text_input("Password", type='password')
    if st.sidebar.checkbox("Login"):
        create_usertable()
        result = login_user(username,password)
        if result:
            st.success("Logged in as {}".format(username))

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