# fastapi-api.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re
import emoji
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import uvicorn
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')

app = FastAPI(title="Multiclass Sentiment Analysis API") # initialize the api

# Load artifacts
model = joblib.load("sentiment_model.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")
label_encoder = joblib.load("label_encoder.joblib")
top_features = joblib.load("selected_features.joblib")

stop_words = set(stopwords.words("english"))

class ReviewRequest(BaseModel):
    text: str

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = emoji.replace_emoji(text, replace="")
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = " ".join([word for word in word_tokenize(text) if word not in stop_words])
    return re.sub(r"\s+", " ", text).strip()

@app.post("/predict")  
def predict_sentiment(review: ReviewRequest):
    cleaned = clean_text(review.text)
    vectorized = vectorizer.transform([cleaned])
    vectorized_selected = vectorized[:, top_features]
    prediction = model.predict(vectorized_selected)[0]
    sentiment = label_encoder.inverse_transform([prediction])[0]
    return {"sentiment": sentiment}

@app.get("/")   # create get request routepoint
def root():
    return {"message": "Welcome to the Multiclass Sentiment Analysis API!"}

if __name__ == "__main__":
    uvicorn.run("fastapi-api:app", host="0.0.0.0", port=8000, reload=True)
