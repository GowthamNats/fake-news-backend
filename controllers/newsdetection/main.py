import pickle
from controllers.main import News

class NewsDetector(News):
    tfidf = pickle.load(open('./controllers/newsdetection/vectorizer.pkl', 'rb'))
    model = pickle.load(open('./controllers/newsdetection/model.pkl', 'rb'))
    
    def detect_news(self, news):
        transformed_news = self.transform_text(news)
        vector_input = self.tfidf.transform([transformed_news])
        result = self.model.predict(vector_input)[0]

        return "Real News" if result == 1 else "Fake News"