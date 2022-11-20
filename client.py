# client.py
import requests

def test_api(text):
    response = requests.post(
        'http://127.0.0.1:8000/test',
        data={'message':text},
    )
    print(response.json())

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
        data="DE",
    )
    print(response.json())
    
def upload_result():
    result = 0
    response = requests.post(
        'http://127.0.0.1:8000/result',
        data={"result":result},
    )
    print(response.json())
    
#upload_photo()

upload_location()

#upload_result()

#test_api('LOL')