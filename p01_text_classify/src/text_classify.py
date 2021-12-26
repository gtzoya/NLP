# -*- coding: utf-8 -*-
"""
Usage:  text classification
Author: gtzoya
Email:  gtzoya@163.com
Date:   2021-12-26 17:06:00
"""

import pandas as pd
import jieba
import os
import datetime,time
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer

def data_loader(file):
    """
    note  :     used to load text file with seperate symble
    input :     string, filename with path
    output:     texts and labels
    """
    
    df = pd.read_csv(file,dtype=str,encoding='utf-8',header=None,sep='\t')
    texts = df.iloc[:,1]
    labels = df.iloc[:,0]
    time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'{time}:  {file} has been loaded')
    return texts,labels

def load_stop_word(file):
    """
    note  :     used to load stop words
    input :     string, filename with path
    output:     set of  words
    """
    
    stop_words = [word.rstrip() for word in open(file,'r',encoding='utf-8').readlines()]
    stop_words = set(stop_words)
    time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'{time}:  stop words has been loaded')
    return stop_words
stop_words = load_stop_word("../resource/stopwords.txt") 

def text2words(textList):
    """
    note  :     used to convert text into seperated words without stop_words
    input :     list, list of text
    output:     list of processed text from input 
    """
    
    select_word_list = []
    for text in textList:
        words = ' '.join(jieba.lcut(text,cut_all=False,HMM=True))
        wordsLeft = filter(lambda x : x not in stop_words,words)
        select_word_list.append(''.join(wordsLeft).strip())
    time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'{time}:  texts have been converted to words')
    return select_word_list

def words2vec(wordsList):
    """
    note  :     used to convert seperated words to vector by tfidf
    input :     list, list of text of seperated words
    output:     hash matrix, and vectorizer model
    """
    
    vectorizer = CountVectorizer()
    counter = vectorizer.fit_transform(wordsList)
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(counter)
    time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'{time}:  train data have been converted to be vec')
    return tfidf,vectorizer     

def words2vec_4test(wordsList,vectorizer):  
    """
    note  :           used to convert seperated words to vector accoding an exist model
    input :     
        wordsList:    list, list of text of seperated words
        vectorizer:   vectorizer model
    output:           hash matrix, and vectorizer model
    """
    
    counter = vectorizer.transform(wordsList)
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(counter)
    time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'{time}:  test data has been converted to be vec')
    return tfidf,vectorizer  

def train(trainData,labels): 
    """
    note:            used to train a classifying model by supervised machine learning
    input:
        trainData:   list of features
        labels:      list for labels
    output:          trained model for predicting
    """
    
    clf = MultinomialNB()
    clf.fit(trainData, labels)
    return clf

def predict(model,testData):
    """
    note:            used to test a classifying model
    input:
        model:      a classifying model, produced by training
        testData:   list of features
    output:          
        labels:     list of labels
        y_scores:   matrix (n x m) of probability, 
                    n is number of observations
                    m is the number of kinds of labels 
    """
    
    labels = model.predict(testData)
    y_score = model.predict_proba(testData)
    return labels,y_score

def evaluate(y_true,y_pred):
    """
    note:  used  to evaluate the model
    input:
        y_true:   list of labels, which are ground truth
        y_pred:   list of labels, which are predicted by classification model
    output:
        cm:       confusion matrix
        report:   precision and recall of different classes
    """
    
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true,y_pred)
    return cm,report

if __name__ == "__main__":  
    t0 = time.time()
    dataPth = "../data/cnews"
    
    print('\n - - - - - - - files allocation- - - - - -')
    train_file = os.path.join(dataPth,'cnews.train.txt')
    test_file  = os.path.join(dataPth,'cnews.test.txt')
    
#    #    # for developer test
#    train_file = os.path.join(dataPth,'cnews.dev.txt')
#    test_file  = os.path.join(dataPth,'cnews.dev.txt')
    
    print('\n - - - - - - - processing training data- - - - - -')
    text_train,y_train = data_loader(train_file)
    words_train = text2words(text_train)
    X_train,vec0 = words2vec(words_train)
    
    print('\n - - - - - - - processing testing data - - - - - -')    
    text_test, y_test  = data_loader(test_file)
    words_test  = text2words(text_test)
    X_test, vec  = words2vec_4test(words_test,vec0)
    
    
    print('\n - - - - - - - training - - - - - -')
    clf = train(X_train,y_train)  
    
    print('\n - - - - - - - testing - - - - - -')      
    y_pred,y_score = predict(clf,X_test)
    
    print('\n - - - - - - - report - - - - - - - - ')
    cm,report = evaluate(y_test,y_pred)
    print('\n confusion matix:')
    print(cm)
    print('\n precision & recall:')
    print(report)
    t1 = time.time()
    print(f'Total elapse: {t1-t0}')
    print('The End ... ...')
