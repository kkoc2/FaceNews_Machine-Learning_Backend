from flask import Flask , request , json# flask kütüphanemizi projemize import ettik.

import re
import random
import warnings
import mysql.connector as sql
import numpy as np
import pandas as pd

import pymysql

#Sistemde tek seferlik 'nltk' kutuphanesini guncelliyor
#import nltk
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')

random.seed(0)
np.random.seed(0)

def read_DB():
    db_connection = sql.connect(host='localhost', database='ym491', user='root', password='yyy')
    db_cursor = db_connection.cursor()
    db_cursor.execute('SELECT * FROM news')
    table_rows = db_cursor.fetchall()
    columns = [col[0] for col in db_cursor.description]
    df = pd.DataFrame(table_rows,columns=columns)
    db_connection.close()
    return df

def clean_text(text):
    stop_words = stopwords.words('english')
    stemmer = PorterStemmer()
    text = str(text).lower() # lowering
    text = text.encode("ascii", "ignore").decode() # non ascii chars
    text = re.sub(r'\n',' ', text) # remove new-line characters
    text = re.sub(r'\W', ' ', text) # special chars
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text) # single characters
    text = re.sub(r'\^[a-zA-Z]\s+', ' ', text) # single char at first
    text = re.sub(r'[0-9]', ' ', text) # digits
    text = re.sub(r'\s+', ' ', text, flags=re.I) # multiple spaces
    return ' '.join([stemmer.stem(word) for word in word_tokenize(text) if word not in stop_words])

def main_clean_text_GET(df):
    main_text = df[df['text'].notna()]
    main_text['text']=df['text'].apply(lambda x:clean_text(x))
    #print(main_text.head())
    return main_text

def score_DISPLAY_PA(main_clean_text):
    x_train, x_test, y_train, y_test = train_test_split(main_clean_text['text'], main_clean_text['label'])
    tfidf_text = TfidfVectorizer()
    train_x = tfidf_text.fit_transform(x_train)
    test_x = tfidf_text.transform(x_test)
    pac_text = PassiveAggressiveClassifier().fit(train_x, y_train)
    y_pred = pac_text.predict(test_x)
    #print(f"Accuracy : {accuracy_score(y_test, y_pred)}") #Accuracy : 0.9574263147755732
    #print(f"F1-Score : {f1_score(y_test, y_pred)}")  #F1-Score : 0.9567260622674759
    return accuracy_score(y_test, y_pred) , f1_score(y_test, y_pred)

def model_CREATE_PA(main_clean_text):
    tfidf_unigram_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,1))
    PA = PassiveAggressiveClassifier(class_weight='balanced')
    pipeline = Pipeline([('vectorizer', tfidf_unigram_vectorizer), ('classifier', PA)])
    X_train, X_test, Y_train, Y_test = train_test_split(main_clean_text['text'], main_clean_text['label'], random_state=0)
    pipeline.fit(X_train, Y_train)
    joblib.dump(pipeline, 'trained_model_PA.pkl', compress=True)
    #print(f"Model-Score : {pipeline.score(X_test, Y_test)}") #Model-Score : 0.9583895203236371
    #return pipeline.score(X_test, Y_test)

def load_model_PA():
    classifier = joblib.load('trained_model_PA.pkl')
    return classifier

def display_score_PA(newtitle,newtext):
    prediction_mapper = {0:"0", 1:"1"} #{1:"REAL", 0:"FAKE"}
    new_clean_text = clean_text(str(newtitle) + ' ' + str(newtext))
    classifier = load_model_PA()
    predict = classifier.predict(pd.Series(new_clean_text))
    for prediction in predict:
        #return print("Prediction: " + prediction_mapper.get(prediction))
        return  prediction_mapper.get(prediction) , 100

def load_model_LR():
    classifier = joblib.load('trained_model_LR.pkl')
    return classifier

def score_DISPLAY_LR(main_clean_text):
    x_train, x_test, y_train, y_test = train_test_split(main_clean_text['text'], main_clean_text['label'])
    tfidf_text = TfidfVectorizer()
    train_x = tfidf_text.fit_transform(x_train)
    test_x = tfidf_text.transform(x_test)
    lr_text = LogisticRegression(C=1e5).fit(train_x, y_train)
    y_pred = lr_text.predict(test_x)
    #print(f"Accuracy : {accuracy_score(y_test, y_pred)}") #Accuracy : 0.9574263147755732
    #print(f"F1-Score : {f1_score(y_test, y_pred)}")  #F1-Score : 0.9567260622674759
    return accuracy_score(y_test, y_pred) , f1_score(y_test, y_pred) 

def model_CREATE_LR(main_clean_text):
    tfidf_unigram_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,2))
    LR = LogisticRegression(C=1e5)
    pipeline = Pipeline([('vectorizer', tfidf_unigram_vectorizer), ('classifier', LR)])
    X_train, X_test, Y_train, Y_test = train_test_split(main_clean_text['text'], main_clean_text['label'], random_state=0)
    pipeline.fit(X_train, Y_train)
    joblib.dump(pipeline, 'trained_model_LR.pkl', compress=True)
    #print(f"Model-Score : {pipeline.score(X_test, Y_test)}") #Model-Score : 0.9583895203236371
    #return pipeline.score(X_test, Y_test)

def display_score_LR(newtitle,newtext):
    prediction_mapper = {0:"0", 1:"1"} #{1:"REAL", 0:"FAKE"}
    new_clean_text = clean_text(str(newtitle) + ' ' + str(newtext))
    classifier = load_model_LR()
    predict = classifier.predict(pd.Series(new_clean_text))
    predict_prob = classifier.predict_proba(pd.Series(new_clean_text))
    for prediction in predict:
        #return print("Prediction: " + prediction_mapper.get(prediction))
        if prediction_mapper.get(prediction) == '0': #{1:"REAL", 0:"FAKE"}
            prob = 100 * predict_prob[:, 0]
        else:
            prob = 100 * predict_prob[:, 1]
        return  prediction_mapper.get(prediction) , prob[0]

app = Flask(__name__) # app değişkenizimizin Flask olduğunu belirttik.

@app.route("/fit_PA", methods=["GET"])
def fit_PA():
    data = main_clean_text_GET(read_DB())
    model_CREATE_PA(data)
    return str(score_DISPLAY_PA(data))

@app.route("/predict_PA", methods=['POST'])
def predict_PA():
    data = json.loads(request.data) 
    title = data['title']
    text = data['text']
    return str(display_score_PA(title, text))

@app.route("/fit_LR", methods=["GET"])
def fit_LR():
    data = main_clean_text_GET(read_DB())
    model_CREATE_LR(data)
    return str(score_DISPLAY_LR(data))

@app.route("/predict_LR", methods=['POST'])
def predict_LR():
    data = json.loads(request.data) 
    title = data['title']
    text = data['text']
    return str(display_score_LR(title, text))
