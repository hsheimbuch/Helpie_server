# client.py
import requests

def upload_photo():
    filename = "/Users/henry/Downloads/Telegram Desktop/photo_2022-11-16_23-30-11.jpg"
    files = {'file': (filename, open(filename, 'rb'))}

    response = requests.post(
        'http://127.0.0.1:8000/analyze',
        files=files,
    )
    print(response.json())
    
def upload_location():
    location = "DE"
    response = requests.post(
        'http://127.0.0.1:8000/location',
        data={"location":location},
    )
    print(response.json())
    
def upload_questionaire():
    first_answer = 1
    second_answer = 2
    second_answer = 3
    response = requests.post(
        'http://127.0.0.1:8000/questionaire',
        data={"first_answer":first_answer,"second_answer":second_answer},
    )
    print(response.json())
    
#upload_photo()

#upload_location()

upload_questionaire()