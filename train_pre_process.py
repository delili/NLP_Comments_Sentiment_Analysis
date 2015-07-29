#-*- coding: UTF-8 -*-
from BeautifulSoup import BeautifulSoup
import urllib, re, sqlite3, os
import chardet
import sys
import jieba

reload(sys)
sys.setdefaultencoding('utf-8')

file_r = file("cn_sample_data/sample.positive.txt", "r")

file_pro_w = file("cn_sample_data/pro_positive.txt", "w")

file_stopwords = file("stopwords.txt", "r")
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
    #for i in jiebas:
    #    code = chardet.detect(str(i))['encoding']
    #    i = str(i).decode(code).encode("utf-8")
    #pro_data =  " ".join(list(set(jiebas)-set(stopwords)))
    #pro_data = " ".join(jiebas)
    #new_row = "<review id=\"" + str(review['id']) + "\">\n" + str(pro_data)  + "\n</review>\n"
    #new_row = "<review id=\"" + str(review['id']) + "\">" + str(pro_data)  + "</review>\n"
    new_row = str(pro_data)
    new_data.append(str(new_row).strip())
for r in new_data:
    file_pro_w.write(r)
file_r.close()
file_pro_w.close()
file_stopwords.close()
