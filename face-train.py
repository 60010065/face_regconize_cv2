import os
import cv2
import numpy as np 
from PIL import Image
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#image_dir = os.path.join(BASE_DIR,"images")
recognizer = cv2.face.LBPHFaceRecognizer_create()

face_cascade = cv2.CascadeClassifier('cascade/data/haarcascade_frontalface_alt2.xml')

current_id =0
label_ids={}
y_labels = []
x_trains = []

for root, dirs,files in os.walk(BASE_DIR):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root,file)
            label = os.path.basename(root).replace(" ","-").lower()
            print(label,path)
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            
            id_ =label_ids[label]
            # print(label_ids)

            pill_image = Image.open(path).convert("L")
            size = (550,550)
            fianl_image = pill_image.resize(size,Image.ANTIALIAS)
            image_array = np.array(fianl_image,"uint8")
            # print(image_array)
            faces = face_cascade.detectMultiScale(image_array,scaleFactor=1.5,minNeighbors = 5)

            for (x,y,w,h) in faces:
                roi = image_array[y:y+h,x:x+w]
                x_trains.append(roi)
                y_labels.append(id_)
with open("labels.pickle",'wb') as f:
    pickle.dump(label_ids,f)
recognizer.train(x_trains,np.array(y_labels))
recognizer.save("trainer.yml")