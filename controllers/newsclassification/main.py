import pickle
from controllers.main import News

class NewsClassifier(News):
    tfidf = pickle.load(open('./controllers/newsclassification/vectorizer.pkl', 'rb'))
    model = pickle.load(open('./controllers/newsclassification/model.pkl', 'rb'))
    categorical_labels = ['Technology', 'Sports', 'World', 'Politics', 'Entertainment', 'Automobile', 'Science']

    def classify_news(self, news):
        transformed_news = self.transform_text(news)
        vector_input = self.tfidf.transform([transformed_news])
        result = self.model.predict(vector_input)[0]

        return self.categorical_labels[result]