import numpy as np
import pandas as pd
import re

from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import pickle

def clean_data(data):
    data = str(data).lower()
    data = re.sub(r"http\S+|https\S+","", data)
    data = re.sub(r"[^a-z\s\']", "", data)
    data = re.sub(r"\@w+|\#","", data)
    data = re.sub(r"\w*\d\w*", "", data)
    data = re.sub(r"\[.*?\]", "", data)
    data = re.sub(r"<.*?>+", "", data)
    data = re.sub(r'(?<=\s)[\'"]+(?=\s)|^[\'"]+|[\'"]+$', '', data)
    filtered_data = [word for word in data.split() if not word in stopwords.words("english")]
    return " ".join(filtered_data)

def train_model():
    df = pd.read_csv('labeled_data.csv')
    df.drop(columns=['Unnamed: 0'], inplace=True)
    speech_class = []
    speech = []
    for i in range(df['count'].count()):
        hate_speech_percentage = df['hate_speech'].iloc[i] / df['count'].iloc[i]
        offensive_language_percentage = df['offensive_language'].iloc[i] / df['count'].iloc[i]
        neither_percentage = df['neither'].iloc[i] / df['count'].iloc[i]
        max_index = np.argmax(np.array([hate_speech_percentage, offensive_language_percentage, neither_percentage]))
        speech_class.append(max_index)
        speech.append(clean_data(df['tweet'][i]))
    speech_class = np.array(speech_class)
    speech = np.array(speech)
    speech_train, speech_test, speech_class_train, speech_class_test = train_test_split(speech, speech_class, test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    speech_train_tfidf = vectorizer.fit_transform(speech_train)
    speech_test_tfidf = vectorizer.transform(speech_test)

    #model = LogisticRegression(max_iter = 1000)
    #model = MultinomialNB()
    #model = RandomForestClassifier(n_estimators=200, random_state=42)
    # lr_grid = {'C':[0.001,0.01,0.1,1,10,100]}
    # model_grid = GridSearchCV(SVC(kernel='rbf'), param_grid=lr_grid, cv=5)
    # model_grid.fit(speech_train_tfidf, speech_class_train)
    # model = model_grid.best_estimator_

    model = SVC(kernel='linear', C=1, probability=True)
    fitted_model = model.fit(speech_train_tfidf, speech_class_train)
    with open('model.pkl', 'wb') as file:
        pickle.dump(model, file)
    speech_class_pred = model.predict(speech_test_tfidf)
    report = classification_report(speech_class_test, speech_class_pred)
    accuracy = accuracy_score(speech_class_test, speech_class_pred)

    # print(report)
    # print(accuracy)
    return fitted_model, vectorizer, report, accuracy