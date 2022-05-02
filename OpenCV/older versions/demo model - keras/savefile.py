import numpy as np
import os
import cv2
import easygui

VIDEO_TYPE = {
    '.avi': cv2.VideoWriter_fourcc(*'XVID')}

RESOLUTION = {
    "480p": (640, 480),
    "720p": (1280, 720)}

def get_res(capture, res):
    width, height = RESOLUTION["480p"]
    return width, height

def get_type(filename):
    filename, ext = os.path.splitext(filename)
    if ext in VIDEO_TYPE:
      return  VIDEO_TYPE[ext]
    return VIDEO_TYPE['.avi']


    

 



