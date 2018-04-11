#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import os
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, SimpleRNN,Activation
from keras.layers.embeddings import Embedding
from datetime import datetime
from gensim.models import word2vec

model_fileName = "word2vec.model"

model = word2vec.Word2Vec.load(model_fileName)

model = Sequential()
model.add(Embedding(1000, 64, input_length=10))
model.add(SimpleRNN(unroll=True, units=50))
# model.add(SimpleRNN( output_dim=50, unroll=True, ))

OUTPIT_SIZE = 10
model.add(Dense( OUTPIT_SIZE))
model.add(Activation('softmax'))

# embedding_matrix = np.zeros((len(word_index) + 1, dim))
# for word, i in word_index.items():
#     embedding_vector = embedding_index.get(word)
#     embedding_matrix[i] = embedding_vector
#
# model.compile(optimizer='adam',loss='categorical_crossentropy',  metrics=['accuracy'])

# model.fit(x = train_result, y = train_label, epochs = 500, validation_split = 0.1, batch_size = 5, verbose = 1)
# scores = model.evaluate(x = train_result, y = train_label, batch_size=5)
# res = model.predict(test_feature, batch_size=5, verbose=0)
#
# print(scores)
#
# result_txt = "result" + str(datetime.now()).split()[1] + ".txt"
# ids = 0
# with open(result_txt, 'a') as out:
#     out.write("id,survived" + '\n')
#     for value in res:
#         if value[0] > 0.6:
#             out.write(str(ids) + "," + str(1) + '\n')
#         else:
#             out.write(str(ids) + "," + str(0) + '\n')
#         ids += 1
#
#
