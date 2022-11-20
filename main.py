from fastapi import FastAPI, File, UploadFile, Request, Form
import cv2
from PIL import Image
import numpy as np
import tensorflow as tf
import os
import csv
import shlex
from io import BytesIO
import requests

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
        reader = csv.DictReader(csvfile)
        for row in reader:
            countires_dict[row['Country code']] = row['Organisations'].split('|')
            for entry_index, entry in \
    enumerate(countires_dict[row['Country code']]):
                countires_dict[row['Country code']][entry_index] = \
    shlex.split(entry[1:-1].replace(',','').replace('[','').replace(']','')\
    .replace('Ukrainians Russian speakers','Ukrainians, Russian speakers'))
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

    faces_coordinates = cascade.detectMultiScale(frame, 1.1, 6)
    
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

@app.get('/test')
async def test_api(message: str = Form(...)):
    return {'Hello World': message}

@app.get('/analyze')
async def get_image(file: str = "https://firebasestorage.googleapis.com/v0/b/helpie-fbe77.appspot.com/o/images%2Fexternal%2Fimages%2Fmedia%2F1000002429.jpg?alt=media&token=0c209923-f2f2-4a8e-87ef-b687d3231602"):
    response = requests.get(file)
    image = np.array(Image.open(BytesIO(response.content)))
    face, face_coordinates = detect_face(image,cascade)
    if face is not None:
        idx, num = detect_emotion(preprocess(face),model)
        emotion_number = idx
    elif face is None:
        emotion_number = 7
    #image = np.array(Image.open(file.file))
    return {'emotion': emotion_number}

@app.get('/location')
async def get_location(location: str = "DE"):
    return {'location': countries_dict[location]}

cards_file_path = 'file.csv'

def return_result(cards_file_path, emotion_number):
    result = {}
    with open(cards_file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for id, row in enumerate(reader):
            if id == emotion_number:
                result['title'] = row[2]
                result['description'] = ''
                for index, card in enumerate(row[4:]):
                    if len(card) > 1:
                        result['description'] += card
                
    return result
            
        
#return_result(cards_file_path, 0)

@app.get('/result')
async def get_result(result: int = 0):
    cards = {'cards':return_result(cards_file_path, result)}
    return_dict = {{'emotion': result},cards}
    return return_dict