# -*- coding: utf-8 -*-
from __future__ import print_function
from pydoc import doc
from sklearn import metrics


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC

# import joblib           #savemodel
from sklearn.datasets import load_digits
import array as arr 
import pandas as pd
import normalize_text as nText   #normalize_text, clean hint
import data_shopee              #get_review
import data_source            #getDataSource
import re
import requests


class train():
        #Lấy data và tiền xử lý 
    ds = data_source.getDataSource()
    train_data = pd.DataFrame(ds.load_data('data_clean/train.crash'))
    new_data = []

        #Thêm mẫu bằng cách lấy trong từ điển Sentiment (nag/pos)
    for index,row in enumerate(nText.pos_list):
        new_data.append(['pos'+str(index),'0',row])
    for index,row in enumerate(nText.nag_list):
        new_data.append(['nag'+str(index),'1',row])

    new_data = pd.DataFrame(new_data,columns=list(['id','label','review']))
    train_data.append(new_data)

        #THÊM STOPWORD LÀ NHỮNG TỪ KÉM QUAN TRỌNG
    stop_ws = (u'rằng',u'thì',u'là',u'mà')

    X_train, X_test, y_train, y_test = train_test_split(train_data.review, train_data.label, test_size=0.3,random_state=42)
    X_train, y_train = ds.transform_to_dataset(X_train,y_train)
    X_test, y_test = ds.transform_to_dataset(X_test, y_test)


        #Try some models
    classifiers = [
                # MultinomialNB(),
                # DecisionTreeClassifier(),
                # LogisticRegression(),
                # SGDClassifier(),
                LinearSVC(fit_intercept = True,multi_class='crammer_singer', C=1),
            ]
    for classifier in classifiers:
        steps = []
        steps.append(('CountVectorizer', CountVectorizer(ngram_range=(1,5),stop_words=stop_ws,max_df=0.5, min_df=5)))
        steps.append(('tfidf', TfidfTransformer(use_idf=False, sublinear_tf = True,norm='l2',smooth_idf=True)))
        steps.append(('classifier', classifier))
        clf = Pipeline(steps)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        report1 = metrics.classification_report(y_test, y_pred, labels=[1,0], digits=3)


    X_train, y_train = ds.transform_to_dataset(train_data.review, train_data.label)

        #TRAIN OVERFITTING/ERRO ANALYSIS
    clf.fit(X_train, y_train)

    # filename = 'digits_classifier.joblib.pkl'
    # #         #Save model
    # # _ = joblib.dump(clf, filename, compress=9)
    # clf2 = joblib.load(filename)

    y_pred = clf.predict(X_train)
    report2 = metrics.classification_report(y_train, y_pred, labels=[1,0], digits=3)




    #     #ERRO ANALYSIS
    # for id,x, y1, y2 in zip(train_data.id, X_train, y_train, y_pred):
    #     if y1 != y2:
    #         # CHECK EACH WRONG SAMPLE POSSITIVE/NAGATIVE
    #         if y1!=1:#0:
    #             print(id,x, y1, y2)

        #CROSS VALIDATION
    cross_score = cross_val_score(clf, X_train,y_train, cv=5)

        #REPORT
    print('DATASET LEN %d'%(len(X_train)))
    print('TRAIN 70/30 \n\n',report1)
    print('TRAIN OVERFITING\n\n',report2)
    print("CROSSVALIDATION 5 FOLDS: %0.4f (+/- %0.4f) \n" % (cross_score.mean(), cross_score.std() * 2))

def test_a_link_shopee(a,url):
    if url == "exit": 
        return
       

        #Lấy review trang shopee lưu vào file data_clean/test.crash
    data_shopee.getReview(url)

        #Lấy data từ file
    test_data = pd.DataFrame(a.ds.load_data('data_clean/test.crash', is_train=False))

    test_list = []
    for document in test_data.review:
        document = nText.normalize_text(document)
        test_list.append(document)
        
    y_predict = a.clf.predict(test_list)
    d1, d2 = 0, 0
    for y in zip(y_predict):
        if y[0] == 1:
            d1 = d1 + 1
        if y[0] == 0:
            d2 = d2 + 1
    
    print("\nSố bình luận tích cực: " + str(d2) +"\n")
    print("Số bình luận tiêu cực: " + str(d1) +"\n")
    print("Tỉ lệ tích cực: %0.4f \n" %( d2 / (d1 + d2) ))
    test_data['label'] = y_predict

    #####Sắp xếp output trên file submit.csv theo thứ tự tích cực, tiêu cực
    # test_data['content'] = test_list
    # test_data = test_data.sort_values(by=['label'])
    test_data[['id', 'label']].to_csv('submit.csv', index=False)
    a= [d2,d1,d2 / (d1 + d2)]
    return a
# a=train()
# test(a,'https://shopee.vn/M%C5%A9-b%E1%BA%A3o-hi%E1%BB%83m-3-4-BALDER-light-%C4%91en-nh%C3%A1m-i.115491915.8112351912?sp_atk=da41bd88-eca9-4b0a-8245-6d20b99eb26a')

def test_a_sentence(a,sentence):
    f = open("data_clean/test.crash", "r+", encoding="utf-8")
    contents = f.read().split("\n")
    f.seek(0)                     
    f.truncate()
    i = 0
    number = []
    comment = []
    number.append("test_" + str(1).zfill(6))
    comment.append("\""+str(sentence) + str(" " + "\"\n"))
    f.write(number[i] + '\n' + comment[i] + '\n')
    f.close()


    test_data = pd.DataFrame(a.ds.load_data('data_clean/test.crash', is_train=False))

    test_list = []
    for document in test_data.review:
        document = nText.normalize_text(document)
        test_list.append(document)
    print(test_list)
    y_predict = a.clf.predict(test_list)
    d1, d2 = 0, 0
    for y in zip(y_predict):
        if y[0] == 1:
            d1 = d1 + 1
        if y[0] == 0:
            d2 = d2 + 1

    return d2

    