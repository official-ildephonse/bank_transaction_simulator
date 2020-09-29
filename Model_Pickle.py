# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 16:16:32 2020

@author: Nithin
"""


#Importing Relevant Libraries
from flask import Flask,request
import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.linear_model import SGDClassifier
from bs4 import BeautifulSoup
import re #import regex expression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier
import pickle
import streamlit as st
Replace_Symbol_with_Space = re.compile("[/(){}\[\]|@,;.:-]") #re.compile("[regex expression]") - means compiling a regular expression into an regex object
Replace_Bad_Symbols = re.compile("[ˆ0-9 #+=_]")
Stopwords = set(stopwords.words("english"))   

#@app.route('/predict_file',methods=["POST"])
def retrain_model(df):
    df.dropna(subset = ["tags"], inplace = True) #inplace = False means the dataframe does not exist and is stored in memory unless you assign a new variable to it
    df["post"]= df["post"].apply(cleantext)
    df["post"].apply(lambda x: len(x.split(" "))).sum()
    X = df["post"] #can be wrriten X =df.post
    Y = df["tags"] #can be written Y =df.tags
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.25, random_state = 42)
    sgd = Pipeline([("vect", CountVectorizer()),
               ("tfidf", TfidfTransformer()),
               ("clf", SGDClassifier(loss = "hinge", penalty = "l2", alpha = 0.0001, random_state =42,max_iter =5, tol = None),
               )])
    sgd.fit(X_train, Y_train)
    y_pred = sgd.predict(X_test)
    print("Test"+df.tags.unique()+"END")
    categoryPickle = open("category_pickle.pkl","wb")
    pickle.dump(df.tags.unique(),categoryPickle)
    pickle_out = open("model_classifier.pkl","wb")
    pickle.dump(sgd, pickle_out)
    pickle_out.close()
    score_acc=accuracy_score(Y_test, y_pred)
    confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, float(round(score_acc, 2)), 0.05)
    return ("New model created with an accuracy of " + str(round(score_acc, 2)))

    img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

def cleantext(text):
    """
        text: a string
        
        return: modified initial string
    """

    text = BeautifulSoup(text, "lxml").text #html decoding

    text = str(text).lower() #changing each post into string datatypw and lower case text

    #Remove symbols in each post
    text = Replace_Symbol_with_Space.sub(" ",text)

    #remove bad symbold from each post
    text = Replace_Bad_Symbols.sub(" ",text)

    #delete stopwords from each post
    text = " ".join(word for word in text.split() if word not in Stopwords)
    
    return text




