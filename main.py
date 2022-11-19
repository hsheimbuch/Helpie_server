from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
import cv2
from PIL import Image
import numpy as np
import tensorflow as tf
import os

# Create array with emotion names
class_names = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Change working dir to script location
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# Initialise Haar cascade and TF model
cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
model = tf.keras.models.load_model("model.h5")

# Function to detect emotions using sliced face image 
# and TF model in .h5 format
def detect_emotion(frame, model):

    emotion = list(model.predict(tf.expand_dims((frame), axis=0)))
    num = max(emotion[0])
    idx = list(emotion[0]).index(num)

    return idx, num

# Function to normalise and resize image 
# prior to processing with TF
def preprocess(frame):

    frame = cv2.resize(frame, (48, 48))
    frame = frame / 255.

    return frame


# Function to detect a single biggest
# face on the image
def detect_face(frame, cascade):
    
    face = None

    faces_coordinates = cascade.detectMultiScale(frame, 1.1, 10)
    
    largest_face_coordinates = None
    
    if len(faces_coordinates) == 0:
        print("No face detected!")
        return face, (0,0,0,0)
    
    for (x, y, w, h) in faces_coordinates:
        
        if (largest_face_coordinates == None) or \
            (w+h > (largest_face_coordinates[2] + largest_face_coordinates[3])):
            largest_face_coordinates = (x, y, w, h)
            
    x, y, w, h = largest_face_coordinates
        
    face = frame[y:y+h, x:x+w]

    return face, (x,y,w,h)

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome from the API"}


@app.post("/analyze")
def get_image(file: UploadFile = File(...)):
    print('a')
    image = np.array(Image.open(file.file))
    face, face_coordinates = detect_face(image,cascade)
    if face is not None:
        idx, num = detect_emotion(preprocess(face),model)
        class_name = class_names[idx]
    else:
        class_name = None
    #image = np.array(Image.open(file.file))
    return {"emotion": class_name}