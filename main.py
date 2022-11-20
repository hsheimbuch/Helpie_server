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
        emotion_number = idx
    else:
        emotion_number = 7
    #image = np.array(Image.open(file.file))
    return {'emotion': emotion_number}

@app.post('/location')
def get_location(location: str = Form(...)):
    return {'location': countries_dict[location]}

@app.post('/result')
def get_result(result: int = Form(...)):
    return {'emotion': 'Worry',
            'description': 'Practice to help you calm \
down in a stressful situation',
            'time': '5',
            'cards':{
            '1': 'During this exercise, \
you will observe the work of your consciousness,\
imagining that it is a white room through which thoughts pass.\
You can perform it in any quiet place, sitting or lying down.\
Close your eyes and take a few deep breaths.\
Breathe slowly and evenly throughout the exercise.',
            '2': 'Imagine that you are in a medium-sized white room \
with two doors. Thoughts enter through one door and \
leave through another. As soon as a thought appears, \
concentrate on it and try to categorize it as evaluative \
or non-evaluative (Example of an evaluative thought: \
“I will look stupid at tomorrow’s performance, they \
will laugh at me” / Example of a non-judgmental one: \
“I am afraid of tomorrow’s performance, how can I \
anxious .. ”note by the author of the channel)',
            '3': 'Consider each thought carefully, with curiosity \
and compassion until it goes away. Don’t try to analyze it, \
just note if it’s evaluative or not. Don’t challenge it, don’t \
try to believe or disbelieve in it. Just be aware that this is \
a thought, a brief moment of your brain activity, an \
occasional visitor to your white room.',
            '4': 'Beware of thoughts that you have classified as evaluative. \
They will try to take possession of you, to force you to \
accept the assessment. The point of this exercise is to \
notice how “sticky” judgmental thoughts are—how they get \
stuck in your mind and how difficult it is to get rid of them. \
You will determine that a thought is painful and judgmental by \
how long it stays in the white room, or by whether you begin \
to feel any emotion about it.',
            '5': 'Try to constantly maintain even breathing, keep a clear \
image of the room and doors, follow thoughts and classify \
them. Remember that a thought is just a thought. \
You are much bigger than her. You are the one who \
creates the white room through which thoughts are allowed \
to pass. You have a million of them, they leave, but you \
still remain. Thought does not require any action from you.\
A thought does not oblige you to believe in it. \
Thought is not you.',
            '6': 'Just watch them walk through the white room.\
Let them live their short life and tell yourself that they \
have a right to exist, even estimated ones.',
            '7': ' Just acknowledge your thoughts, let them go when the time \
comes, and get ready to meet new ones one by one. Keep doing \
this exercise until you feel that you have truly distanced \
yourself from your thoughts.  Do it until even evaluative \
thoughts begin to pass through the room without lingering. \
P.S.  Instead of the image of a room with 2 doors, you can \
take an image that is closer to you: for example, a funicular \
that transfers thoughts or a baggage belt at an airport, etc.'
            }}