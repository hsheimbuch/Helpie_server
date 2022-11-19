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
    
upload_photo()