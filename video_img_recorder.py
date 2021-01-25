import numpy as np 
import cv2
import os

filename = 'video.avi'
frame_per_seconds = 24.0
my_res = '720p'

def change_res(cap,width,heigth):
    cap.set(3,width)
    cap.set(4,heigth)

STD_DIMENSIONS= {
    "480p": (640,480),
    "720p": (1280,720),
    "1080p": (1920,1080),
    "4K": (3840,2160),
}

def get_dims(cap,res='1080p'):
    width,heigth = STD_DIMENSIONS['480p']
    if res in STD_DIMENSIONS:
        width,heigth =STD_DIMENSIONS[res]
        change_res(cap,width,heigth)
        return width,heigth

VIDEO_TYPE ={
    'avi':cv2.VideoWriter_fourcc(*'XVID'),
    'mp4':cv2.VideoWriter_fourcc(*'XVID'),
}

def get_video_type(filename):
    filename,ext = os.path.splitext(filename)
    if ext in VIDEO_TYPE:
        return VIDEO_TYPE[ext]
    return VIDEO_TYPE['avi']


cap = cv2.VideoCapture(0)
dim = get_dims(cap,res=my_res)
video_type_cv2 = get_video_type(filename)

out = cv2.VideoWriter(filename,video_type_cv2,frame_per_seconds,dim)


while True:
    ret,frame = cap.read()

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    out.write(frame)
    cv2.imshow('frame',frame)

    #cv2.imshow('gray',gray)

    if cv2.waitKey(20) & 0xFF  == ord('q'):
        break
cap.release()
out.release
cv2.destroyAllWindows()