#! /usr/bin/python
# -*- coding:utf-8 -*-

from __future__ import division
from collections import defaultdict
import collections
import numpy as np
import random
import nltk
import gensim
import math
import os

from gensim import corpora, models, similarities

def preprocess(doc):
    doc_sen = nltk.sent_tokenize(doc)
    english_punctuations = ['\'s','\'','\"',',', '.', ':', ';', '?', '(', ')', '[', ']', '!', '@', '#', '%', '$', '*',':',';','...']
    predoc = []
    for sen in doc_sen:
        sen_word = [word for word in nltk.word_tokenize(sen) if word not in english_punctuations]
        predoc.append(sen_word)
    return predoc

def build_dict(docs):
    docs_word = []
    for doc in docs:
        for sen in doc:
            docs_word = docs_word + sen
    word_set = set(docs_word)
    dictionary = dict()
    for word in word_set:
        dictionary[word] = len(dictionary)
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reversed_dictionary

def doc_tran(docs,dictionary,reversed_dictionary):
    POS_dictionary = dict()
    for POS in ['CC','CD','DT','EX','FW','IN','JJ','JJR','JJS','LS','MD','NN','NNP','NNPS','NNS','PDT','POS','PRP','PRP$','RB','RBR','RBS','RP','SYM','TO','UH','VB','VBD','VBG','VBN','VBP','VBZ','WDT','WP','WP$','WRB',':','\'\'']:
        POS_dictionary[POS] = len(POS_dictionary)
    POS_NUM = len(POS_dictionary)
    Tran_Docs = []
    for doc in docs:
        tran_doc = []
        for sen in doc:
            tran_sen = []
            tag_sen = nltk.pos_tag(sen)
            for word,pos in tag_sen:
                tran_sen.append((dictionary[word],POS_dictionary[pos]))
            tran_doc.append(tran_sen)
        Tran_Docs.append(tran_doc)
    return Tran_Docs, POS_NUM, POS_dictionary


def im_doc_tran(docs, dictionary, reversed_dictionary):
    POS_dictionary = dict()
    for POS in ['NOUN', 'ADV', 'VERB', 'ADJ', 'ELSE']:
        POS_dictionary[POS] = len(POS_dictionary)
    POS_NUM = len(POS_dictionary)
    Tran_Docs = []
    for doc in docs:
        tran_doc = []
        for sen in doc:
            tran_sen = []
            tag_sen = nltk.pos_tag(sen)
            for word,pos in tag_sen:
                if pos in ['NN', 'NNS', 'NNP', 'NNPS']:
                    tran_sen.append((dictionary[word],POS_dictionary['NOUN']))
                elif pos in ['RB', 'RBR', 'RBS']:
                    tran_sen.append((dictionary[word],POS_dictionary['ADV']))
                elif pos in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
                    tran_sen.append((dictionary[word],POS_dictionary['VERB']))
                elif pos in ['JJ', 'JJR', 'JJS']:
                    tran_sen.append((dictionary[word],POS_dictionary['ADJ']))
                else:
                    tran_sen.append((dictionary[word],POS_dictionary['ELSE']))
            tran_doc.append(tran_sen)
        Tran_Docs.append(tran_doc)
    return Tran_Docs, POS_NUM, POS_dictionary

def initial(Docu_NUM,Topic_NUM,Embedding_SIZE,POS_NUM,dictionary,docs):
# Topic INIT for all sent in all doc
    Topic_List = []
    m = np.zeros((Docu_NUM,Topic_NUM))
    n = np.zeros((len(dictionary),Topic_NUM))
    doc_count = 0
    tao = np.random.random(POS_NUM)
    tao[0] = 0.1
    i_word = np.zeros(len(dictionary))
    for doc in docs:
        topic_doc = []
        for sen in doc:
            ran = random.random()
            temp_k = int(ran*Topic_NUM)
            topic_doc.append(temp_k)
            m[doc_count,temp_k] = m[doc_count,temp_k] + 1
            for word in sen:
                n[word[0],temp_k] = n[word[0],temp_k] + 1
                i_word[word[0]] = np.random.binomial(1,tao[word[1]])
        Topic_List.append(topic_doc)
        doc_count = doc_count + 1
#    print i_word
# Initial topic vector
    Topic_Vec = np.random.random((Topic_NUM, Embedding_SIZE))
# Initial word vector
    Word_Vec = np.zeros((len(dictionary),Embedding_SIZE))
    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',binary = True)
    for word in dictionary:
        try:
            Word_Vec[dictionary[word]] = model.wv[word]
        except:
            Word_Vec[dictionary[word]] = np.random.random(Embedding_SIZE)
    return m,n,tao,Topic_List, Topic_Vec, Word_Vec,i_word

def Update(m,n,tao, Topic_List,Topic_Vec,Word_Vec,i_word,docs,beta,alpha):
    for d in range(len(docs)):
        for s in range(len(docs[d])):
            s_topic = Topic_List[d][s]
            m[d,s_topic] = m[d,s_topic] - 1
            for word,_ in docs[d][s]:
                n[word,s_topic] = n[word,s_topic] - 1
            P_z = np.zeros(Topic_NUM)
            for k in range(Topic_NUM):
                temp_prod = 0
                for word,pos in docs[d][s]:
                    temp_prod = temp_prod + ((1-tao[pos]) * (n[word,k] + beta)/(sum(n[:,k])+len(dictionary)*beta) +  tao[pos] * np.exp((Word_Vec[word]+Topic_Vec[s_topic]).dot(Word_Vec[word])/(Word_Vec[word].dot(Word_Vec[word]))))
#                    print np.exp((Word_Vec[word]+Topic_Vec[s_topic]).dot(Word_Vec[word])) 
#                    print np.exp((Word_Vec[word]+Topic_Vec[s_topic]).dot(Word_Vec[word])/(Word_Vec[word].dot(Word_Vec[word]))) 
#                    print (1-tao[pos]) * (n[word,k] + beta)/(sum(n[:,k])+len(dictionary)*beta) 
#                    print 'l'
#                    print tao[pos]
                P_z[k] = temp_prod + np.log(m[d,k] + alpha)
            P_z = P_z/np.sum(P_z)
            s_topic = np.random.multinomial(1,P_z)
            s_topic = list(s_topic).index(1)
            for word,pos in docs[d][s]:
                P_i_1 = tao[pos] * np.exp((Topic_Vec[s_topic]).dot(Word_Vec[word])/Word_Vec[word].dot(Word_Vec[word])) * 0.01
                P_i_0 = (1-tao[pos]) * (n[word,s_topic] + beta)/((sum(n[:,s_topic]) + len(dictionary) * beta)*(np.sqrt(n[:,s_topic].dot(n[:,s_topic]))/len(n[:,s_topic])))
#                print '1'
#                print P_i_1
#                print P_i_0
                P_i = P_i_1/(P_i_1 + P_i_0)
                i_word[word] = np.random.binomial(1,P_i)
            m[d,s_topic] = m[d,s_topic] + 1
            Topic_List[d][s] = s_topic
            for word,_ in docs[d][s]:
                n[word,s_topic] = n[word,s_topic] +1
    return m,n,tao,Topic_List, Topic_Vec, Word_Vec,i_word

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def Embedding_Update(m,n,tao,Topic_List,Topic_Vec, Word_Vec, i_word, docs, eta,neg):
    for d in range(len(docs)):
        for s in range(len(docs[d])):
            for word_index in range(len(docs[d][s])):
                if i_word[docs[d][s][word_index][0]]==1:
                    if word_index != 0 and word_index != len(docs[d][s])-1:
                        C_w = []
                        C_w.append(docs[d][s][word_index-1][0])
                        C_w.append(docs[d][s][word_index+1][0])
                    if word_index ==0:
                        C_w = []
                        C_w.append(docs[d][s][word_index+1][0])
                    if word_index == len(docs[d][s])-1:
                        C_w = []
                        C_w.append(docs[d][s][word_index-1][0])
                    neg_sample = np.random.random(neg)*len(i_word)
                    neg_sample = int(neg_sample)
                    for negs in range(neg_sample):
                        x_w = Word_Vec[docs[d][s][word_index][0]] + Topic_Vec[Topic_List[d][s]] 
                        Topic_Vec[Topic_List[d][s]] = Topic_Vec[Topic_List[d][s]] + eta * (0.01-sigmoid(x_w.dot(Word_Vec[negs])-np.log(neg/len(i_word)))) * Word_Vec[negs]
                    for cw in C_w:
                        x_w = Word_Vec[docs[d][s][word_index][0]] + Topic_Vec[Topic_List[d][s]] 
                        Topic_Vec[Topic_List[d][s]] = Topic_Vec[Topic_List[d][s]] + eta * (0.99-sigmoid(x_w.dot(Word_Vec[cw])-np.log(neg/len(i_word)))) * Word_Vec[cw]
    return m,n,tao,Topic_List, Topic_Vec,Word_Vec,i_word

def Update_tao(m,n,tao,Topic_List,i_word,docs,POS_NUM,POS_dictionary):
    sum_tao = np.zeros(POS_NUM)
    sum_tao_i = np.zeros(POS_NUM)
    print len(i_word) - sum(i_word)
    for i in range(len(docs)):
        for sen in docs[i]:
            for word,pos in sen:
                sum_tao[pos] = sum_tao[pos] + 1
                if i_word[word] == 1:
                    sum_tao_i[pos] = sum_tao_i[pos] + 1
    sum_tao_i = sum_tao_i + 1
    sum_tao = sum_tao + POS_NUM
#    print 'i'
#    print sum_tao_i
#    print 't'
#    print sum_tao
    tao = sum_tao_i/sum_tao
    #print tao
    return tao

def readdocu(path):
    files = os.listdir(path)
#    for filedir in files:
#        filedoc = os.listdir(path+'/'+filedir)
    document = []
    print('path is ',path)
    for doc in files:
        if doc != '.DS_Store':
            with open(path+'/'+doc) as f:
                print('doc is ',doc)
                docu_num = ''
                while f.readline() != '<TEXT>\n':
                    pass
                line = f.readline()
                line = line.lower()
                while line != '</TEXT>\n':
                    line = line.lower()
                    docu_num = docu_num + ' ' + line.strip()
                    line = f.readline()
                document.append(docu_num)
    return document

def evalue(docs,n,m,Topic_List,alpha,beta,Docu_NUM,Topic_NUM,Word_NUM):
    phi = np.zeros((Topic_NUM,Word_NUM))
    tphi = np.zeros((Topic_NUM,Word_NUM))
    v = np.zeros((Docu_NUM,Topic_NUM))
    tv = np.zeros((Docu_NUM,Topic_NUM))
    word_co = np.zeros((Docu_NUM,Word_NUM))
    for d in range(Docu_NUM):
        for s in range(len(docs[d])):
            s_topic = Topic_List[d][s]
            for word,_ in docs[d][s]:
                if i_word[word] == 0:
                    v[d][s_topic] = v[d][s_topic] + 1
                phi[s_topic][word] = phi[s_topic][word] + 1
                word_co[d,word] = word_co[d,word] + 1
    for i in range(Topic_NUM):
        for j in range(Word_NUM):
            tphi[i][j] = (phi[i][j] + beta)/(np.sum(phi[i,:]) + Word_NUM * beta)
    for i in range(Docu_NUM):
        for j in range(Topic_NUM):
            tv[i][j] = (v[i][j] + alpha)/(np.sum(v[i,:]) + Topic_NUM * alpha)
    per = 0
    for i in range(Docu_NUM):
        temp_p = 0
        for j in range(Word_NUM):
            temp_pp = 0
            for k in range(Topic_NUM):
                temp_pp = temp_pp + tphi[k,j] * tv[i,k]
            temp_p = temp_p + word_co[i,j] * np.log(temp_pp)
        per = per - temp_p
    per = per/(np.sum(word_co))
    per = np.exp(per)
    return per

    
def TfIdf_Sim(doc_pre):
    dictionary = corpora.Dictionary(doc_pre)
    corpus = [dictionary.doc2bow(text) for text in doc_pre]
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    sim = []
    index = similarities.MatrixSimilarity(corpus_tfidf)
    for doc_sim in index:
        sim.append(doc_sim)
    sim = np.asarray(sim)
    return sim


if __name__ == "__main__":
    path = 'd04a'
    docs = readdocu(path)
    history = 'Bush calls on power to dispense largesse. '
    docs[0] = history + docs[0]
    Docu_NUM = len(docs)
    for i in range(Docu_NUM):
        docs[i] = preprocess(docs[i])
    dictionary,reversed_dictionary = build_dict(docs)
    Word_NUM = len(dictionary)
    Tran_Docs, POS_NUM, POS_dictionary = im_doc_tran(docs,dictionary,reversed_dictionary)
    Topic_NUM = 30
    Embedding_SIZE = 300
    m,n,tao,Topic_List,Topic_Vec,Word_Vec,i_word = initial(Docu_NUM,Topic_NUM,Embedding_SIZE,POS_NUM,dictionary,Tran_Docs)
    Iteration_NUM = 20
    eta = 0.1
    neg = 1
    beta = 0.01
    alpha = 50/Topic_NUM
    for itera in range(Iteration_NUM):
        m,n,tao,Topic_List,Topic_Vec,Word_Vec,i_word = Update(m,n,tao,Topic_List,Topic_Vec,Word_Vec,i_word,Tran_Docs,beta,alpha)
        m,n,tao,Topic_List,Topic_Vec,Word_Vec,i_word = Embedding_Update(m,n,tao,Topic_List,Topic_Vec,Word_Vec,i_word,Tran_Docs,eta,neg)
        tao = Update_tao(m,n,tao,Topic_List,i_word,Tran_Docs,POS_NUM,POS_dictionary)
        per = evalue(Tran_Docs,n,m,Topic_List,alpha,beta,Docu_NUM,Topic_NUM,Word_NUM)
        print per
    chose_topic = Topic_List[0][0]
    chose_doc = []
    for i in range(Docu_NUM):
        for j in range(len(Topic_List[i])):
            if Topic_List[i][j] == chose_topic:
                chose_doc.append(docs[i][j])
    sim = TfIdf_Sim(chose_doc)
    max_sim = 0
    sim_index = 0
    for i in range(1,len(chose_doc)):
        if sim[i].dot(sim[0])/(np.linalg.norm(sim[i]) * np.linalg.norm(sim[0])) > max_sim:
            max_sim = sim[i].dot(sim[0])/(np.linalg.norm(sim[i]) * np.linalg.norm(sim[0]))
            sim_index = i
    print chose_doc[sim_index]





    
def movie_LTE(path, history):
    docs = []
    sentence = open(path).readlines()
    for s in range(len(sentence)):
        temp_s = sentence[s].split(" +++$+++ ")
        sentence[s] = temp_s[-1] 
    pre_sen = preprocess(sentence)
    pre_history = preprocess(history)
    docs = [pre_sentence, pre_history]
    Docu_NUM = len(docs)
    dictionary,reversed_dictionary = build_dict(docs)
    Word_NUM = len(dictionary)
    Tran_Docs, POS_NUM, POS_dictionary = im_doc_tran(docs,dictionary,reversed_dictionary)
    Topic_NUM = 10
    Embedding_SIZE = 300
    m,n,tao,Topic_List,Topic_Vec,Word_Vec,i_word = initial(Docu_NUM,Topic_NUM,Embedding_SIZE,POS_NUM,dictionary,Tran_Docs)
    Iteration_NUM = 100
    eta = 0.1
    neg = 1
    beta = 0.01
    alpha = 50/Topic_NUM
    for itera in range(Iteration_NUM):
        m,n,tao,Topic_List,Topic_Vec,Word_Vec,i_word = Update(m,n,tao,Topic_List,Topic_Vec,Word_Vec,i_word,Tran_Docs,beta,alpha)
        m,n,tao,Topic_List,Topic_Vec,Word_Vec,i_word = Embedding_Update(m,n,tao,Topic_List,Topic_Vec,Word_Vec,i_word,Tran_Docs,eta,neg)
        tao = Update_tao(m,n,tao,Topic_List,i_word,Tran_Docs,POS_NUM,POS_dictionary)
    chose_doc = []
    chose_topic = Topic_List[1,0]
    for i in len(Topic_List[0]):
        if Topic_List[0,i] == chose_topic:
            chose_doc.append(sentence[i])
    chose_doc.insert(0,history[0])
    sim = TfIdf_Sim(chose_doc)
    max_sim = 0
    sim_index = 0
    for i in range(1,len(chose_doc)):
        if sim[i].dot(sim[0])/(np.linalg.norm(sim[i]) * np.linalg.norm(sim[0])) > max_sim:
            max_sim = sim[i].dot(sim[0])/(np.linalg.norm(sim[i]) * np.linalg.norm(sim[0]))
            sim_index = i
    return sentence[sim_index]


#if __name__ == "__main__":
#    path = 'NLP/src/data/movie_lines.txt'
#    history = ['Like my fear of wearing pastels?']
#    print movie_LTE(path, history)
