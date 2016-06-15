from __future__ import division
from collections import Counter
import numpy as np
import cPickle
import sys
import random
import theano
import theano.tensor as T

SENTENCE_START_TOKEN = "START"
SENTENCE_END_TOKEN = "END"
UNKNOWN_TOKEN = "UNKNOWN"

def save(path, obj):
    with open(path, 'wb') as f:
        cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)

def load_data(filename='./data/new_val_train.txt', vocabulary_size=20000, min_sent_characters=0):
    word_to_index = []
    index_to_word = []
    word_counter = Counter()

    sentences = []
    question = []

    for line in open(filename, 'r'):
        if line == '\n': continue
        pair = line.split('\t')
        sent = pair[1]
        que = pair[0]
        sents = sent.strip().split()
        ques = que.strip().split()
        sentences.append( [SENTENCE_START_TOKEN] + sents + [SENTENCE_END_TOKEN])
        question.append( [SENTENCE_START_TOKEN] + ques + [SENTENCE_END_TOKEN])
        word_counter.update(sents)
        word_counter.update(ques)
    print 'the numbers of sentences is : %d' % len(sentences)
    print 'the numbers of question is : %d' % len(question)

    vocab_count = word_counter.most_common(vocabulary_size)
    vocab = {SENTENCE_START_TOKEN:1, SENTENCE_END_TOKEN:2, UNKNOWN_TOKEN:0}
    for i, (word, count) in enumerate(vocab_count):
        vocab[word] = i + 3
    print 'the size of vocabulary: %d' % len(vocab)
    word_to_index = vocab
    index_to_word = {value:key for key, value in word_to_index.iteritems()}

    Y = [[word_to_index.get(w,0) for w in sent[:-1]] for sent in sentences]
    Y_idx = [[word_to_index.get(w,0) for w in sent[1:]] for sent in sentences]
    X = [[word_to_index.get(w,0) for w in sent[:-1]] for sent in question]
    
    i = 0
    X_Y_idx = []
    for sen in X:
        x_train = np.asarray(sen).astype('int32')
        y_train = np.asarray(Y[i]).astype('int32')
        yidx_train = np.asarray(Y_idx[i]).astype('int32')
        X_Y_idx.append([x_train, y_train, yidx_train])
        i = i+1

    #save dict and trainset
    save('./data/train.pkl',X_Y_idx)
    save('./data/dict.pkl', word_to_index)

    return X_Y_idx, word_to_index

load_data()


