import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm

data = pd.read_csv('training_data.csv')

X = data['text']
y = data['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

clf = svm.SVC()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print('Model accuracy:', accuracy)

new_text = ['This product is amazing, I highly recommend it']
new_text_vectorized = vectorizer.transform(new_text)
sentiment = clf.predict(new_text_vectorized)
print('Sentiment:', sentiment)