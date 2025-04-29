# train.py

import pandas as pd
import re
import emoji
import joblib
import numpy as np
import nltk
from langdetect import detect, DetectorFactory, LangDetectException
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import chi2
from sklearn.utils.class_weight import compute_class_weight

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

DetectorFactory.seed = 0
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
custom_stopwords = {"would", "shall", "could", "might"}
stop_words.update(custom_stopwords)
stop_words.discard("not")

def is_english(text):
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False

def clean_text(text):
    text = text.lower()
    text = emoji.replace_emoji(text, replace="")
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = " ".join([
        lemmatizer.lemmatize(word)
        for word in word_tokenize(text)
        if word not in stop_words
    ])
    return re.sub(r"\s+", " ", text).strip()

def prepare_data():
    df = pd.read_csv("aliexpressdata.csv")
    df = df.rename(columns={'REVIEW_CONTENT': 'content', 'RATING': 'rating'})
    df = df[['content', 'rating']].drop_duplicates().dropna()
    df = df[df["content"].apply(is_english)]
    df["cleaned_review"] = df["content"].apply(clean_text)
    df["sentiment"] = df["rating"].apply(lambda x: "Positive" if x >= 4 else "Neutral" if x == 3 else "Negative")
    return df

def train_and_save_model():
    df = prepare_data()
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=10000, min_df=10)
    X = vectorizer.fit_transform(df["cleaned_review"])
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["sentiment"])

    # Feature selection
    chi2_scores, _ = chi2(X, y)
    top_features = np.argsort(chi2_scores)[-5000:]
    X_selected = X[:, top_features]

    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, stratify=y, random_state=42)

    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

    model = LogisticRegression(max_iter=2000, class_weight=weight_dict, solver="liblinear")
    model.fit(X_train, y_train)

    joblib.dump(model, "sentiment_model.joblib")
    joblib.dump(vectorizer, "tfidf_vectorizer.joblib")
    joblib.dump(label_encoder, "label_encoder.joblib")
    joblib.dump(top_features, "selected_features.joblib")
    print("Model, vectorizer, label encoder, and top features saved.")

if __name__ == "__main__":
    train_and_save_model()
