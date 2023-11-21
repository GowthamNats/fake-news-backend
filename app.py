from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from controllers.newsdetection.main import NewsDetector
from controllers.newsclassification.main import NewsClassifier
from controllers.newssentiment.main import NewsSentiment
from controllers.newscrawler.main import NewsCrawler

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

# Initialize the classes for usage
detector = NewsDetector()
classifier = NewsClassifier()
sentiment = NewsSentiment()
crawler = NewsCrawler()

# Base root to check
@app.get('/')
def hello_world() -> dict:
  return {
    "message": "Hello World"
  }

# News Detector route
@app.post('/detect')
async def news_detection(news: dict) -> dict:
  return {
    "prediction": detector.detect_news(news["query"])
  }

# News Classifier route
@app.post('/classify')
async def news_classification(news: dict) -> dict:
  return {
    "prediction": classifier.classify_news(news["query"])
  }

# News Sentiment route
@app.post('/sentiment')
async def news_sentiment(news: dict) -> dict:
  return {
    "prediction": sentiment.news_sentiment(news["query"])
  }

# News Crawler route
@app.post('/crawl')
async def crawl_news(news: dict) -> dict:
  return {
    "prediction": crawler.crawl_news(news["query"])
  }

# Consolidated Route
@app.post("/analyze")
async def analyze_news(news: dict) -> dict:
  value = detector.detect_news(news["query"])
  if value == "Fake News":
    return {
      "detection": value,
      "classification": None,
      "sentiment": None,
      "crawl": None
    }
  else:
    return {
      "detection": value,
      "classification": classifier.classify_news(news["query"]),
      "sentiment": sentiment.news_sentiment(news["query"]),
      "crawl": crawler.crawl_news(news["query"])
    }