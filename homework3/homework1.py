#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import os
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from datetime import datetime
from gensim.models import word2vec

times = str(datetime.now()).split(".")[0]
datetime = times.split()[0].replace("-", "_")
clocktime = times.split()[1].replace(":", "_")
model_fileName = "word2vec.model"
# print(model_fileName)

class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def getSentences(self):
        words_cut = []
        for rootDir in os.listdir(self.dirname):
            if rootDir.endswith('_cut') :
                print(self.dirname + "/"+ rootDir)

                for fname in os.listdir(self.dirname+ "/" + rootDir):
                    for line in open(os.path.join(self.dirname, rootDir, fname)):
                        line = ''.join(line.split())
                        words_cut.append( line.split("/") )
        return words_cut

# mySentences = MySentences('./data/training')
# sentences = mySentences.getSentences()

# model = word2vec.Word2Vec(sentences,  sg=0, size=100,  window=5,  min_count=5,  negative=3, sample=0.001, hs=1, workers=4)
# model.save(model_fileName)

model = word2vec.Word2Vec.load(model_fileName)

res = model.most_similar("日本", topn=100)

print("日本")
print("相似詞前 100 排序:")

for v in res:
    print(v[0], ",", v[1])

print('\r\n\n')

res1 = model.similarity('日本', '台灣')
print('日本', '台灣 : ', res1)
