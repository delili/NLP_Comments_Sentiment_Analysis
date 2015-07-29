#-*- coding: UTF-8 -*-
"""
Author:deli
File:process.py
Time:2015/4/03 11:33:46
Fork: https://github.com/abromberg/sentiment_analysis_python
"""
import re, math, collections, itertools, os
import nltk, nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
import sys
from BeautifulSoup import BeautifulSoup 
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from nltk.collocations import BigramCollocationFinder
from sklearn.neighbors import KNeighborsClassifier   

reload(sys)
sys.setdefaultencoding('utf-8')

RT_POLARITY_POS_FILE = 'cn_sample_data/pos.txt'
RT_POLARITY_NEG_FILE = 'cn_sample_data/neg.txt' 
#处理换行符
def pro_line():
    file_r = file("cn_sample_data/pro_positive.txt", "r")
    file_w = file("cn_sample_data/pos.txt", "w")
    pos_content = file_r.read().decode("utf-8")
    posWords = pos_content.split(" ")
    for w in posWords:
        w.encode("utf-8").strip()
        w = re.sub(r"[0-9a-zA-Z]", "", w)
        if len(w) == 0 or w.isspace():
            continue
        file_w.write(w+" ")
    file_r.close()
    file_w.close()
#获取文档内所有词
def achieve_words():
    posWords = []
    negWords = []
    file_pos = file("cn_sample_data/pos.txt", "r")
    file_neg = file("cn_sample_data/neg.txt", "r")
    pos_content = file_pos.read()
    neg_content = file_neg.read()
    posWords = pos_content.split(" ")
    negWords = neg_content.split(" ")
    file_pos.close()
    file_neg.close()
    return (posWords, negWords)
    

#利用卡方检验做特征选择，假设：words与类别不相关，则得分越大，说明越相关
def create_word_scores(posWords, negWords):
    file_scores = file("cn_sample_data/scores.txt", "w")
    #迭代，将多个序列合并
    
    word_fd = FreqDist()
    cond_word_fd = ConditionalFreqDist()
    for word in posWords:
        word_fd[str(word)] += 1 
        cond_word_fd['pos'][str(word)] += 1
    for word in negWords:
	    word_fd[str(word)] += 1
	    cond_word_fd['neg'][str(word)] += 1
    pos_word_count = cond_word_fd['pos'].N()
    neg_word_count = cond_word_fd['neg'].N()
    total_word_count = pos_word_count + neg_word_count
    word_scores = {}
    for word, freq in word_fd.iteritems():
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][str(word)], (freq, pos_word_count), total_word_count)
        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][str(word)], (freq, neg_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score
    sorted(word_scores.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
    for key in word_scores:
        file_scores.write(str(key)+" : " + str(word_scores[str(key)])+ "\n")
    file_scores.close()
    return word_scores 

def create_word_bigram_scores(posWords, negWords):
    bigram_finder = BigramCollocationFinder.from_words(posWords)
    bigram_finder = BigramCollocationFinder.from_words(negWords)
    posBigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 2000)
    negBigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 2000)

    pos = posWords + posBigrams #词和双词搭配
    neg = negWords + negBigrams

    word_fd = FreqDist()
    cond_word_fd = ConditionalFreqDist()
    for word in pos:
        word_fd[str(word)] += 1 
        cond_word_fd['pos'][str(word)] += 1
    for word in neg:
	    word_fd[str(word)] += 1
	    cond_word_fd['neg'][str(word)] += 1

    pos_word_count = cond_word_fd['pos'].N()
    neg_word_count = cond_word_fd['neg'].N()
    total_word_count = pos_word_count + neg_word_count

    word_scores = {}
    for word, freq in word_fd.iteritems():
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count)
        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score

    return word_scores

def achieve_features(feature_select):
    posFeatures = []
    negFeatures = []
    testFeatures = []
    file_pos = file("test_data/pro_label_pos.txt", "r")
    content = file_pos.read().decode("utf-8")
    soup = BeautifulSoup(content)
    all_review = soup.findAll('review')
    for item in all_review:
        posSentences = str(item.string.encode("utf-8")).strip()
        posWords = posSentences.split(" ")
        posWords = [feature_select(posWords), 'pos']
        posFeatures.append(posWords)
    file_pos.close()
    
    file_neg = file("test_data/pro_label_neg.txt", "r")
    content = file_neg.read().decode("utf-8")
    soup = BeautifulSoup(content)
    all_review = soup.findAll('review')
    for item in all_review:
        negSentences = str(item.string.encode("utf-8"))
        negWords = negSentences.split(" ")
        negWords = [feature_select(negWords), 'neg']
        '''for w in negWords:
            print "---------" 
            print type(w) 
            print "---------" 
            fw.write(str(w.keys()))
            print '11111111111'
        '''
        negFeatures.append(negWords)
    file_neg.close()
    trainFeatures = posFeatures + negFeatures
    
    file_test = file("test_data/pro_label_test.txt", "r")
    content = file_test.read().decode("utf-8")
    soup = BeautifulSoup(content)
    all_review = soup.findAll('review')
    for item in all_review:
        testSentences = str(item.string.encode("utf-8")).strip()
        testWords = testSentences.split(" ")
        flag = item['label']
        if flag == '1':
            testWords = [feature_select(testWords), 'pos']
        elif flag == '0':
            testWords = [feature_select(testWords), 'neg']
        testFeatures.append(testWords)
    file_test.close() 
    return (trainFeatures ,testFeatures)
def evaluate(classifier_alo):
    
    classifier = SklearnClassifier(classifier_alo) #在nltk 中使用scikit-learn 的接口
    classifier.train(trainFeatures) #训练分类器
    
    referenceSets = collections.defaultdict(set)
    testSets = collections.defaultdict(set)	
    i = 0
    for item in testFeatures:
        referenceSets[item[1]].add(i)
        predicted = classifier.classify(item[0])
        testSets[predicted].add(i)	
        i += 1
    
    pos_pre = nltk.metrics.precision(referenceSets['pos'], testSets['pos'])
    pos_recall = nltk.metrics.recall(referenceSets['pos'], testSets['pos'])
    neg_pre =  nltk.metrics.precision(referenceSets['neg'], testSets['neg'])
    neg_recall = nltk.metrics.recall(referenceSets['neg'], testSets['neg'])
    
    print (str('{0:.3f}'.format(float(pos_pre))) + "  "
    +str('{0:.3f}'.format(float(pos_recall))) + "  "
    +str('{0:.3f}'.format(float(neg_pre))) + "  "
    +str( '{0:.3f}'.format(float(neg_recall))) + "  "
    +str('{0:.3f}'.format(2*(float(pos_pre)*float(pos_recall)) / (float(pos_recall)+float(pos_pre)))) + "  "
    +str('{0:.3f}'.format(2*(float(neg_pre)*float(neg_recall)) / (float(neg_recall)+float(neg_pre)))))
    
#排序，并反回number个词的set
def find_best_words(word_scores, number):
	best_vals = sorted(word_scores.iteritems(), key=lambda (w, s): s, reverse=True)[:number]
	best_words = set([w for w, s in best_vals])
	return best_words

#仅用选出的特征词来判断该条评论的正负
def best_word_features(words):
	return dict([(word, True) for word in words if word in best_words])

if __name__ == '__main__':
    #pro_line()
    posWords = []
    negWords = []
    posFeatures = []
    negsFeatures = []
    (posWords, negWords) = achieve_words()
    #word_scores = create_word_bigram_scores(posWords, negWords)
    word_scores = create_word_scores(posWords, negWords)
    dimension = ['100','500','800','1500','5000']
    #best_words = find_best_words(word_scores, 100)
    for d in dimension:
        best_words = find_best_words(word_scores, int(d))
        (trainFeatures, testFeatures) = achieve_features(best_word_features)
        
        evaluate(BernoulliNB())
        evaluate(MultinomialNB())
        evaluate(LogisticRegression())
        evaluate(SVC(gamma=0.001, C=100.))
        evaluate(NuSVC())
        evaluate(KNeighborsClassifier())
        print '------------------------------------------'
