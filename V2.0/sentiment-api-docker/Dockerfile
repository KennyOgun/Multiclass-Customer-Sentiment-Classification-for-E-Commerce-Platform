# Dockerfile

# Use an official lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all necessary files
COPY api.py sentiment_model.joblib tfidf_vectorizer.joblib label_encoder.joblib selected_features.joblib ./
COPY requirements.txt ./
# Copy everything to the container
#COPY . .



# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY ./app ./app

# Download necessary nltk resources
RUN python -m nltk.downloader punkt stopwords punkt_tab

# Expose the port FastAPI will run on (FastAPI runs on 8000 by default)
EXPOSE 10000

# Run the API with uvicorn
#CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]


# Run app
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "10000"]

# # Dockerfile

# # Use an official lightweight Python image
# FROM python:3.10-slim

# # Set working directory
# WORKDIR /app

# # Copy all necessary files
# COPY api.py sentiment_model.joblib tfidf_vectorizer.joblib label_encoder.joblib selected_features.joblib ./
# COPY requirements.txt ./
# # Copy everything to the container
# #COPY . .

# # Install dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# # Download necessary nltk resources
# RUN python -m nltk.downloader punkt stopwords punkt_tab

# # Expose the port FastAPI will run on (FastAPI runs on 8000 by default)
# EXPOSE 8000

# # Run the API with uvicorn
# CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
