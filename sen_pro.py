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

