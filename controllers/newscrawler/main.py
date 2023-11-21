from transformers import pipeline
import requests

from controllers.main import News
from config import settings

class NewsCrawler(News):
    keyword_extractor = pipeline("token-classification", model="yanekyuk/bert-uncased-keyword-extractor")

    def crawl_news(self, news):
        tokens = list(set([token["word"] for token in self.keyword_extractor(news)]))
        search_query = "+".join([token for token in tokens if len(token) > 2])
        results = []
        while True:
            parameters = {
                "q": search_query,
                "language": "en",
                "pageSize": 10,
                "apiKey": settings.APIKEY
            }
            response = requests.get("https://newsapi.org/v2/everything", params=parameters).json()
            if response['totalResults'] > 0:
                results = response['articles']
                break
            search_query = search_query[:search_query[::-1].find("+")]

        return results

