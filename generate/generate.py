from __future__ import division
from collections import Counter
import numpy as np
import cPickle
import sys
import rnn as rnned
import random

SENTENCE_START_TOKEN = "START"
SENTENCE_END_TOKEN = "END"
UNKNOWN_TOKEN = "UNKNOWN"

def sample(a, temperature=0.1):
    a = np.array(a, dtype='double')
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a))
def print_sentence(s, index_to_word):
    sentence_str = [index_to_word[x] for x in s[1:-1]]
    print(" ".join(sentence_str))
    sys.stdout.flush()

def generate_sentence(model, index_to_word, word_to_index, c):
    # We start the sentence with the start token
    new_sentence = [word_to_index[SENTENCE_START_TOKEN]]
    while not new_sentence[-1] == word_to_index[SENTENCE_END_TOKEN]:
        next_word_probs = model.predict(c, new_sentence)[-1]
        sampled_word = sample(next_word_probs)
        new_sentence.append(sampled_word)
    print_sentence(new_sentence, index_to_word)

model_path = '../best.mdl'
dict_path = '../data/dict.pkl'
train_path = '../data/train.pkl'

print 'load model...'
best = cPickle.load(open(model_path, "r"))
rParameters = best[0]

#load data
word_to_index = cPickle.load(open(dict_path, "r"))
index_to_word = {value:key for key, value in word_to_index.items()}
train = cPickle.load(open(train_path, 'r'))

#parameters == train.py

nhidden = 50
vobsize = len(word_to_index)
emb_dimension = 50

rnn = rnned.RNNED(nh=nhidden, nc=vobsize, de=emb_dimension, model= rParameters)


dbo = {'A':'1','B':'2','C':'3','D':'4','E':'5','F':'6','G':'7'}

#'''
while True:
    string = raw_input('Enter your conversation: ')
    temp = [dbo.get(w) for w in string.strip().split()]
    x_train = [word_to_index[SENTENCE_START_TOKEN]]+[word_to_index[w] for w in string.strip().split()]

    sentence = [index_to_word[x] for x in x_train[1:]]
    print " ".join(sentence)
    generate_sentence(rnn, index_to_word, word_to_index, x_train)
'''
while True:
    wait = raw_input("please press enter")
    i = random.randint(0, len(train)-2)
    x_train = train[i][1]
    sentence = [index_to_word[x] for x in x_train[1:]]
    print " ".join(sentence)
    c = train[i][0]
    generate_sentence(rnn, index_to_word, word_to_index, c)
'''















