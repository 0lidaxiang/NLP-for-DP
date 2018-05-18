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

#idenity paramaters
word_vec_len = 256


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

# get word embedding vector
answer = word2vec.Word2Vec.load("word2vec_20180430.model")
word_vectors = answer.wv
wvv = word_vectors.vocab
wvv_keys = wvv.keys()
wvv_keys_list = list(wvv_keys)

texts_list = []
for text in train_texts_list:
    texts_list.append(text)
    
for text in test_texts:
    texts_list.append(text)

del word_vectors, wvv, train_texts_list

tokenizer1 = Tokenizer(num_words = 20000)
tokenizer1.fit_on_texts(texts_list)
sequences = tokenizer1.texts_to_sequences(texts_list)  # word index list
max_doc_word_length = 200
sequences1 = pad_sequences(sequences, maxlen=max_doc_word_length, padding='post')

print("Found %s unique tokens" % len(tokenizer1.word_index))

word_index = tokenizer1.word_index
word_counts = tokenizer1.word_counts
num_words = max(max(sequences))

word_index_keys_list = list(word_index.keys())
word_index_values_list = list(word_index.values())
    
docs_index = tokenizer1.texts_to_sequences(texts_list)

vocab_size = 20000 + 1
# create a weight matrix for wordds in training docs
embedding_matrix = np.zeros((vocab_size, word_vec_len))
ss = 0

for word, i in word_index.items():
    if ss < 20000:
        if i < 20000:
            embedding_vector = answer[word]
            embedding_matrix[i] = embedding_vector
    else:
        break

del answer

embedding_layer = Embedding(
                            input_dim= embedding_matrix.shape[0],
                            output_dim= 256,
                            weights=[embedding_matrix], 
                            input_length = 200,
                            trainable=False)
model = Sequential()
model.add(embedding_layer )
model.add(Conv1D(256, 3,padding = 'same', ))
model.add(Flatten())
model.add(Dense(2, activation='relu'))
model.compile(optimizer='adam',loss='mae')
model.summary()


Y_label_list = np.zeros((36000, 2))
for ids in range(0, 36000):
    Y_label_list[ids][0] = label_list[ids][0]
    Y_label_list[ids][1] = label_list[ids][1]

history = model.fit(x = sequences1[0:36000], y = Y_label_list,  batch_size= 10000,  epochs = 150, verbose = 1)

np_loss_history = np.array(loss_history)
np.savetxt("loss_history.txt", np_loss_history, delimiter=",")

fig = plt.figure()
ax = plt.axes()
x = np.linspace(0, 1, 50)
loss_history = np.loadtxt("loss_history.txt")
plt.plot(x, loss_history, '-g');  # dotted red


X_train =sequences1

# evaluate the model
loss_accuracy = model.evaluate(X_train[0:100], Y_label_list[0:100], verbose=1)
print(type(loss_accuracy), loss_accuracy)

test_sequences1 = X_train[36000:40000]

predict_res = model.predict(test_sequences1, batch_size= 3600, verbose=0)

final_res = []
for pre_res in predict_res:
    final_res.append(pre_res)

result_txt = "server_result001" + ".txt"
ids = 0
with open(result_txt, 'w') as out:
    out.write("id,good,bad" + '\n')
    for value in final_res:
        out.write(str(ids) + "," + str(int(value[0])) + "," + str(int(value[1])) + '\n')
        ids += 1

