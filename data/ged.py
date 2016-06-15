import numpy as np
import cPickle
import sys
import random

def save(path, obj):
    with open(path, 'wb') as f:
        cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
e = ['A','B','C','D','E','F','G']
n = ['1', '2', '3', '4','5','6','7']
    
sentences = []
for i in range(50):
    con = []
    res = []
    for j in range(8):
        k = random.randint(0, 6)
        con.append(e[k])
        res.append(n[k])
    sent = " ".join(con) + '\t' + " ".join(res) + '\n'
    sentences.append(sent)
f = file('./new_val_test.txt','w')
f.writelines(sentences)
f.close()

