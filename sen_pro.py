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

def readdoc(path):
    path = 'bbc'

def generate_batch(batch_size, style_list,train_set):
    batch = np.ndarray(shape=(batch_size), dtype = np.int32)


def nce_loss_compute(embed, train_labels, batch_size, nce_num):
    


if __name__ == "__main__":
    doc = readdoc(path)
    for sen in doc:
        sen = senten_tag(sen)
    pro_doc, word_dic, ne_dic, word_reversed_dic, O_value = diction_set(doc)
    style_list = build_style_list(pro_doc, O_value)
    train_set = build_train_set(style_list)
    style_size = len(style_list)
    embedding_size = 300
    batch_size = 100
    num_sampled = 64
    Word_Vec = np.zeros((len(word_dic), embedding_size))
    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary = True)
    for word in word_dic:
        try:
            Word_Vec[word_dic[word]] = model.wv[word]
        except:
            Word_Vec[word_dic[word]] = np.zeros(embedding_size)
    graph = tf.Graph()
    with graph.as_defalut():

        train_inputs = tf.placeholder(tf.int32, shape = [batch_size])
        train_labels = tf.placeholder(tf.int32, shape = [batch_size, 1])

        with tf.device('/cpu:0'):

            embeddings = tf.Variable(tf.random_uniform([style_size, embedding_size], -1.0, 1.0))
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)


            loss = nce_loss_compute(embed, train_labels, batch_size, nce_num)




            nce_weights = tf.Variable(tf.truncated_normal([style_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
            nce_biases = tf.Variable(tf.zeros([style_size]))

            loss = tf.reduce_mean(tf.nn.nce_loss(weights = nce_weights, biaes=nce_biases, labels=train_labels, inputs=embed, num_sampled=num_sampled, num_classes=style_size))

            optimizer = tf.train.GradientDesentOptimizer(1.0).minimize(loss)


        num_steps = 100001

        with tf.Session(graph=graph) as session:

            init.run()
            print("Initialized")

            average_loss = 0

            for step in xrange(num_steps):
                batch_inputs, batch generate
                feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

                _, loss_val = session.run([optimizer, loss], feed_dict = feed_dict)

                average_loss += loss_val

                if step % 2000 == 0:
                    if step > 0:
                        average_loss /= 2000
                    print('Average loss at step ', step, ': ', average_loss)

        final_embeddings = normalized_embeddings.eval()



    

#def generate_batch(batch_size, train_set):
#    batch = 

            

