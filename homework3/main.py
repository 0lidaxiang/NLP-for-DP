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
