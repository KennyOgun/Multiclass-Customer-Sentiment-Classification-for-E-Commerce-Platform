import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import re
import emoji
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from lime.lime_text import LimeTextExplainer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

# Global NLP tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
custom_stopwords = {"would", "shall", "could", "might"}
stop_words.update(custom_stopwords)
stop_words.discard("not")

# Load Models
model = joblib.load("sentiment_model.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")
label_encoder = joblib.load("label_encoder.joblib")
top_features = joblib.load("selected_features.joblib")

# Functions
def clean_text(text):
    if not isinstance(text, str):
        return ""
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

def feature_engineering(df):
    df['cleaned_review'] = df['content'].apply(clean_text)
    df['sentiment'] = df['rating'].apply(lambda x: "Positive" if x >= 4 else "Neutral" if x == 3 else "Negative")
    return df

# App Interface
st.title("💬 Sentiment Analysis App")
df = pd.read_csv("aliexpressdata.csv").rename(columns={'REVIEW_CONTENT': 'content', 'RATING': 'rating'})
df = df[['content', 'rating']].sample(frac=1, random_state=42)
df = feature_engineering(df)

st.subheader("Data Preview")
st.dataframe(df.head())
st.dataframe(df.shape)

st.subheader("Sentiment Distribution")
fig, ax = plt.subplots()
sns.countplot(x='sentiment', data=df, ax=ax)
st.pyplot(fig)

# Transform features
X = vectorizer.transform(df['cleaned_review'])[:, top_features]
y = label_encoder.transform(df['sentiment'])
y_pred = model.predict(X)
y_prob = model.predict_proba(X)

# Evaluation
st.subheader("Model Evaluation")
st.text(classification_report(y, y_pred))
st.write("**ROC AUC Score:**", round(roc_auc_score(y, y_prob, multi_class='ovo', average='macro'), 3))

# Confusion Matrix
cm = confusion_matrix(y, y_pred)
fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
st.pyplot(fig_cm)

# LIME
st.subheader("Model Interpretation with LIME")
user_input = st.text_area("Enter a review to explain:")
if user_input:
    clean_input = clean_text(user_input)
    explainer = LimeTextExplainer(class_names=label_encoder.classes_)
    explanation = explainer.explain_instance(clean_input, lambda x: model.predict_proba(vectorizer.transform(x)[:, top_features]), num_features=6)
    st.components.v1.html(explanation.as_html(), height=500, scrolling=True)
