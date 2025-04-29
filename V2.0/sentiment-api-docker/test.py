# test.py

import requests

url = "http://localhost:8000/predict"
data = {"text": "The product quality is amazing and delivery was fast!"}

response = requests.post(url, json=data)

print("Status Code:", response.status_code)
print("Prediction:", response.json())
