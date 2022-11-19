from fastapi import FastAPI, File, UploadFile, Request, Form
import cv2
from PIL import Image
import numpy as np
import tensorflow as tf
import os
import csv
import shlex

# Change working dir to script location
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# Create array with emotion names
emotion_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
extended_emotion_names = ['Worry', 'Anxiety', 'Panic', 'Melancholy', 'Sadness',
                          'Grief', 'Annoyance', 'Anger', 'Rage', 'Shame', 'Guilt', 
                          'Disgust', 'Neutral', 'Surprised', 'Happy']

# Initialise Haar cascade and TF model
cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
model = tf.keras.models.load_model("model.h5")

# Function to init countries
def init_countries(countries_file):
    countires_dict = {}
    with open(countries_file, newline='') as csvfile:
        spamreader = csv.DictReader(csvfile)
        for row in spamreader:
            countires_dict[row['Country code']] = row['Organisations'].split('|')
            for entry_index, entry in \
    enumerate(countires_dict[row['Country code']]):
                countires_dict[row['Country code']][entry_index] = \
    shlex.split(entry[1:-1].replace(',','').replace('[','').replace(']','')\
    .replace('Ukrainian Russian speakers','Ukrainians, Russian speakers'))
    return countires_dict

# Init countries
countries_file = 'countires.csv'
countries_dict = init_countries(countries_file)

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
        print('No face detected!')
        return face, (0,0,0,0)
    
    for (x, y, w, h) in faces_coordinates:
        
        if (largest_face_coordinates == None) or \
            (w+h > (largest_face_coordinates[2] + largest_face_coordinates[3])):
            largest_face_coordinates = (x, y, w, h)
            
    x, y, w, h = largest_face_coordinates
        
    face = frame[y:y+h, x:x+w]

    return face, (x,y,w,h)

app = FastAPI()

@app.get('/')
def read_root():
    return {'message': 'Welcome from the API'}

@app.post('/test')
def test_api(message: str = Form(...)):
    return {'Hello World': message}

@app.post('/analyze')
def get_image(file: UploadFile = File(...)):
    image = np.array(Image.open(file.file))
    face, face_coordinates = detect_face(image,cascade)
    if face is not None:
        idx, num = detect_emotion(preprocess(face),model)
        class_name = emotion_names[idx]
    else:
        class_name = None
    #image = np.array(Image.open(file.file))
    return {'emotion': class_name}

@app.post('/location')
def get_location(location: str = Form(...)):
    return {'location': countries_dict[location]}

@app.post('/result')
def get_result(result: int = Form(...)):
    return {'result': 'some big text'}