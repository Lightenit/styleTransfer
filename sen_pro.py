#! /usr/bin/python
# -*- coding:utf-8 -*-

from __future__ import division
from nltk.chunk import tree2conlltags
from nltk import word_tokenize, ne_chunk, pos_tag

def senten_tag(sentence):
    ne_tree = ne_chunk(pos_tag(word_tokenize(sentence)))
    iob_tagged = tree2conlltags(ne_tree)
    return iob_tagged


def diction_set(doc):
    docs_word = []
    docs_ne = []
    for sen in doc:
        for ite in sen:
            docs_word.append(ite[0])
            docs_ne.append(ite[-1])
    word_set = set(docs_word)
    ne_set = set(docs_ne)
    word_dic = dict()
    ne_dic = dict()
    for word in word_set:
        word_dic[word] = len(word_dic)
    for ne in ne_set:
        ne_dic[ne] = len(ne_dic)
    pro_doc = []
    for sen in doc:
        pro_sen = []
        for ite in sen:
            pro_sen.append((word_dic[ite[0]] , ne_dic[ite[-1]]))
        pro_doc.append(pro_sen)
    word_reversed_dic = dict(zip(word_dic.values(), word_dic.keys()))
    return pro_doc, word_dic, ne_dic, word_reversed_dic

def style

