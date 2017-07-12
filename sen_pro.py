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
    O_tag = u'O'
    for sen in doc:
        pro_sen = []
        for ite in sen:
            pro_sen.append((word_dic[ite[0]] , ne_dic[ite[-1]]))
        pro_doc.append(pro_sen)
    word_reversed_dic = dict(zip(word_dic.values(), word_dic.keys()))
    return pro_doc, word_dic, ne_dic, word_reversed_dic, ne_dic[O_tag]

def style_extract(sen, O_value):
    style = []
    target = []
    for i in range(len(sen)):
        if sen[i][-1] == O_value:
            style.append(sen[i][0])
        else:
            target.append(sen[i][0])
            style.append(-1)
    return style, target


class sen_style:
    def __init__(self, sen, style, tar):
        self.style = style
        self.placenum = style.count(-1)
        self.placedic = dict()
        self.target = []
        self.target.append(tar)
        for i in range(len(sen)):
            if style[i] == -1:
                self.placedic[i] = [sen[i][-1]]
        self.count = 1
    
    def style_eq(self, style):
        if self.style == style:
            return 1
        else return 0

    def style_add(self, sen, tar):
        self.count = self.count + 1
        self.target.append(tar)
        for i in range(len(sen)):
            if self.style[i] == -1:
                if sen[i][-1] not in self.placedic[i]:
                    self.placedic[i].append(sen[i][-1])


def build_style_list(doc, O_value):
    style_list = []
    for sen in doc:
        style, tar = style_extract(sen, O_value)
        if style_list == []:
            sensty = sen_style(sen, style, tar)
            style_list.append(sensty)
        else:
            style_flag = 0
            for sen_sty in style_list:
                if sen_sty.style_eq(style):
                    sen_sty.add(sen, tar)
                    style_flag = 1
                    break
            if style_flag == 0:
                sensty = sen_style(sen, style, tar)
                style_list.append(sensty)
    return style_list

def build_train_set(style_list):
    train_set = []
    for i in range(len(style_list)):
        for j in range(len(style_list[i].target)):
            train_set.append([i]+style_list[i].target[j])
    return train_set

if __name__ == "__main__":
    

#def generate_batch(batch_size, train_set):
#    batch = 

            

