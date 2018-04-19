#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten

from gensim.models import word2vec
model = word2vec.Word2vec(sentences, size=250)
model.save("word2vec.model")

model.most_similar()
model.similarity(x,y)

model = Sequential()
model.add(embedding_layer)
model.add(SimpleRNN( output_dim=50, unroll=True, ))
model.add(Dense( OUTPIT_SIZE))
model.add(Activation('softmax'))

embedding_matrix = np.zeros((len(word_index) + 1, dim))
for word, i in word_index.items():
    embedding_vector = embedding_index.get(word)
    embedding_matrix[i] = embedding_vector

model.compile(optimizer='adam',loss='categorical_crossentropy',  metrics=['accuracy'])
