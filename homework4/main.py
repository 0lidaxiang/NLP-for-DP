# coding: utf-8
import time
import os
import numpy as np
import keras
from keras import metrics
from keras.utils import to_categorical
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout, Flatten, Conv2D, Conv1D, MaxPooling1D, MaxPooling2D,Activation
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.layers.embeddings import Embedding
from datetime import datetime
from gensim.models import word2vec
from collections import *
import pandas as pd
import matplotlib.pyplot as plt

def text_to_index(corpus):
    new_corpus = []
    for doc in corpus:
        new_doc = []
        for word in doc:
            try:
                new_doc.append(word2idx[word])
            except:
                new_doc.append(0)
        new_doc_arr = np.array(new_doc)
        new_corpus.append( new_doc_arr)
    return np.array(new_corpus)


# get data
# get texts data
category2idx = {'AllTogether': 0, 'Baseball': 1, 'Boy-Girl': 2, 'C_chat':  3, 'CVS': 4,
                  'GameSale': 5, 'GetMarry': 6, 'Lifeismoney': 7, 'LoL': 8, 'MH': 9, 'MLB': 10, 'Mobilecomm': 11, 
                'movie': 12,'MuscleBeach':  13, 'NBA': 14,  'SENIORHIGH': 15, 'Stock': 16, 
                'Tennis': 17, 'Tos': 18, 'WomenTalk': 19}

train_df_sample = pd.read_pickle('train.pkl').sample(frac=1, random_state=123)
train_texts = train_df_sample.values
label_list = train_df_sample.label

test_pickle_df = pd.read_pickle('test.pkl')
test_texts = test_pickle_df["text"].values

train_texts_list = []
for text in train_texts:
    train_texts_list.append(text[0])
    
texts_list = []
for text in train_texts_list:
    texts_list.append(text)
    
for text in test_texts:
    texts_list.append(text)


# get word embedding vector
answer = word2vec.Word2Vec.load("word2vec_20180430.model")
word_vectors = answer.wv
wvv = word_vectors.vocab
wvv_keys = wvv.keys()
wvv_keys_list = list(wvv_keys)
vocab_num = len(wvv.items()) + 1
vocab_list = [(word, word_vectors[word]) for word, _ in wvv.items()]

del word_vectors, wvv, train_texts_list, answer

word_vec_len = 256
embedding_matrix = np.zeros((vocab_num , word_vec_len))
word2idx = {}

for i, vocab in enumerate(vocab_list):
    word, vec = vocab
    embedding_matrix[i + 1] = vec
    word2idx[word] = i + 1

embedding_matrix.shape

embedding_layer = Embedding( input_dim= embedding_matrix.shape[0],output_dim= 256, weights=[embedding_matrix], 
                            input_length = 200,trainable=False)
model = Sequential()
model.add(embedding_layer )
model.add(Conv1D(256, 3,padding = 'same', ))
model.add(Flatten())
model.add(Dense(2, activation='relu'))
model.compile(optimizer='adam',loss='mae')
model.summary()

max_doc_word_length = 200

x_trains_texts = train_df_sample.text.append(test_pickle_df.text)
X_train_texts = text_to_index(x_trains_texts)
X_train = pad_sequences(X_train_texts, maxlen= max_doc_word_length)

Y_label_list = np.zeros((36000, 2))
for ids in range(0, 36000):
    Y_label_list[ids][0] = label_list[ids][0]
    Y_label_list[ids][1] = label_list[ids][1]
print(Y_label_list.shape)

history = model.fit(x = X_train[0:36000], 
                    y = Y_label_list, 
                    batch_size= 10000,
                    epochs = 60, verbose = 1)

loss_history = np.loadtxt("loss_history.txt")
fig = plt.figure()
ax = plt.axes()
x = np.linspace(0, 1, loss_history.shape[0])

plt.plot(x, loss_history, '-g');  # dotted red


model.save('my_model.h5') 

# evaluate the model
loss_accuracy = model.evaluate(X_train[0:100], Y_label_list[0:100], verbose=1)
print(type(loss_accuracy), loss_accuracy)

test_sequences1 = X_train[36000:40000]

predict_res = model.predict(test_sequences1, batch_size= 3600, verbose=0)

final_res = []
for pre_res in predict_res:
    final_res.append(pre_res)

print(predict_res[0])
print(len(final_res))
result_txt = "local_result001" + ".txt"
ids = 0
with open(result_txt, 'w') as out:
    out.write("id,good,bad" + '\n')
    for value in final_res:
        out.write(str(ids) + "," + str(int (round(value[0]))) + "," + str(int (round(value[0]))) + '\n')
        ids += 1

