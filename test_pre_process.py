#-*- coding: UTF-8 -*-
from BeautifulSoup import BeautifulSoup
import urllib, re, sqlite3, os
import chardet
import sys
import jieba
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


reload(sys)
sys.setdefaultencoding('utf-8')

file_r = file("../cn_sample_data/sample.positive.txt", "r")

file_label_w = file("pro_label_pos.txt", "w")

file_stopwords = file("../stopwords.txt", "r")
content = file_r.read().decode('utf-8')
soup = BeautifulSoup(content)
all_review = soup.findAll('review')
stopwords = [line.strip() for line in file_stopwords.readlines()] 
new_data = list()
for review in all_review:
    pro_data = ""
    raw_data = str(review.string)
    words = list(jieba.cut(raw_data, cut_all = False))
    for w in words: 
        seg = str(w.encode('utf-8'))
        if seg not in stopwords:
             pro_data += str(seg)+" " 
    #new_row = "<review id=\"" + str(review['id']) + "\" label=\"" + review['label']+ "\">" + str(pro_data)  + "</review>\n"
    new_row = "<review id=\"" + str(review['id']) + "\">\n" + str(pro_data).strip()  + "\n</review>\n"
    new_data.append(str(new_row))
for r in new_data:
    file_label_w.write(r)
file_r.close()
file_label_w.close()
file_stopwords.close()
