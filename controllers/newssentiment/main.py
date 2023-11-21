from transformers import pipeline
from controllers.main import News

class NewsSentiment(News):
    sentiment_pipe = pipeline("text-classification", model="shashanksrinath/News_Sentiment_Analysis")

    def news_sentiment(self, news):
        return self.sentiment_pipe(news)[0]["label"]