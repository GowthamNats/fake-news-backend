import uvicorn 
from fastapi import FastAPI
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pickle
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('punkt')
nltk.download('stopwords')

# Pre-requisites
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))
ps = PorterStemmer()

def transform_text(text):
  text = text.lower()
  text = nltk.word_tokenize(text)

  y = []
  for i in text:
    if i.isalnum():
      y.append(i)
  text = y[:]
  y.clear()

  for i in text:
    if i not in stopwords.words('english') and i not in string.punctuation:
      y.append(i)
  text = y[:]
  y.clear()

  for i in text:
    y.append(ps.stem(i))

  return " ".join(y)

# FastAPI app generation
app = FastAPI()

origins = [
    "*"
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Base root to check
@app.get('/')
def hello_world() -> dict:
  return {
    "message": "Hello World"
  }

# Predict route for text
@app.post('/predict')
def predict_news(news: dict) -> dict:
    transformed_news = transform_text(news["message"])
    vector_input = tfidf.transform([transformed_news])
    result = model.predict(vector_input)[0]

    if result == 1:
      prediction = "Real News"
    else:
      prediction = "Fake News"

    return {
      "prediction": prediction
    }

# Predict route with OCR

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)