
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


#!/usr/bin/python3
# -*- coding: UTF-8 -*-
import time

import os
import numpy as np
# import keras
from keras import metrics
from keras.utils import to_categorical
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, SimpleRNN, GRU,Activation
from keras.layers.embeddings import Embedding
from datetime import datetime
from gensim.models import word2vec
import pandas as pd


# In[73]:


# get texts data
category2idx = {'Japan_Travel': 0, 'KR_ENTERTAIN': 1, 'Makeup': 2, 'Tech_Job':  3, 'WomenTalk': 4,
                  'babymother': 5, 'e-shopping': 6, 'graduate': 7, 'joke': 8, 'movie': 9}

train_df_sample = pd.read_pickle('train.pkl').sample(frac=1, random_state=123)
# train_df_sample = pd.concat([train_pickle_df.text]).sample(frac=1)


# In[125]:


# get word embedding vector
answer = word2vec.Word2Vec.load("word2vec_20180424.model")
# print(type(answer))
word_vectors = answer.wv
wvv = word_vectors.vocab
wvv_keys = wvv.keys()
wvv_keys_list = list(wvv_keys)
# print(wvv_keys_list[:10]) #['櫻花林', '好比', '考科', '床上', '一點現', '記住', '寶寶的', '柔嫩', '不規則', '朴智妍']


# In[97]:


train_texts = train_df_sample.values
label_list = to_categorical(train_df_sample.category)

test_pickle_df = pd.read_pickle('test.pkl')
test_texts = test_pickle_df["text"].values


# In[135]:


def text_to_index(corpus):
    new_corpus = []
    for doc in corpus:
        new_doc = []
        for word in doc:
            try:
                new_doc.append(word_index[word])
            except:
                new_doc.append(0)
#         new_doc_arr = np.array(new_doc).reshape(1, max_doc_word_length)
        new_doc_arr = np.array(new_doc)
        new_corpus.append( new_doc_arr)
    return np.array(new_corpus)


# In[99]:


train_texts_list = []
for text in train_texts:
    train_texts_list.append(text[0])

# train_texts_index = train_texts_list

print(len(train_texts_list), train_texts_list[0])


# In[100]:


# train_texts_list1 = list(train_texts_list)
# print(type(train_texts_list), len(train_texts_list), type(train_texts_list[0]))
# print(train_texts_list[0])


# In[101]:


tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_texts_list)
sequences = tokenizer.texts_to_sequences(train_texts_list)
# max_doc_word_length = max(len(l) for l in train_texts)
max_doc_word_length = 200
sequences1 = pad_sequences(sequences, maxlen=max_doc_word_length, padding='post')
word_index = tokenizer.word_index
print("Found %s unique tokens" % len(word_index))


# data = pad_sequences(sequences)
# print("Shape of data tensor:" , data.shape)


# In[127]:


vocab_size = len(word_index) + 1
# create a weight matrix for words in training docs
answer_vector_size = answer.vector_size
embedding_matrix = np.zeros((vocab_size, answer_vector_size))
for word, i in word_index.items():
    if word in wvv_keys_list:
        embedding_vector = answer[word]
        if embedding_vector is not None:
            embedding_matrix[i+1] = embedding_vector
#     else:
#         embedding_matrix[i] = np.zeros((1, answer_vector_size))


# In[103]:


# del answer


# In[128]:


embedding_layer = Embedding(input_dim= embedding_matrix.shape[0],
                            output_dim= embedding_matrix.shape[1],
                            weights=[embedding_matrix], 
                            trainable=False)
model = Sequential()
model.add(embedding_layer)
model.add(GRU(16))    
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='softmax'))
# early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=0, verbose=0, mode='auto')
model.compile(optimizer='adam',loss='categorical_crossentropy',  metrics=["accuracy"])
model.summary()


# In[136]:


X_train_texts = text_to_index(train_df_sample.text)
X_train = pad_sequences(X_train_texts, maxlen= max_doc_word_length)


# In[137]:


# X_train_texts_ = list(X_train_texts)
print(len(X_train), type(X_train))
print(len(X_train[0]), type(X_train[0]))
print(X_train[0].shape)
print(X_train[1].shape)


# In[138]:


history = model.fit(x = X_train, y = label_list, 
                    validation_split=0.1, 
                    batch_size= 3000,
                    epochs = 100, verbose = 1)


# In[139]:


# loss_func_name = 'categorical_crossentropy'
# for value in history.history["categorical_crossentropy"]:
#     print(history)
model.save('my_model.h5') 


# In[140]:


# evaluate the model
loss_accuracy = model.evaluate(sequences1[0:100], label_list[0:100], verbose=1)
print(type(loss_accuracy), loss_accuracy)


# In[146]:


# test_tokenizer = Tokenizer()
# test_tokenizer.fit_on_texts(test_texts)
# test_sequences = test_tokenizer.texts_to_sequences(test_texts)
# test_sequences1 = pad_sequences(test_sequences, maxlen=max_doc_word_length, padding='post')

Y_sequences1 = text_to_index(test_pickle_df.text)
Y_sequences11 = pad_sequences(Y_sequences1, maxlen=max_doc_word_length)


# In[148]:


predict_res = model.predict(Y_sequences11, verbose=1)
# print(len(predict_res), predict_res)

final_res = []
for pre_res in predict_res:
    final_res.append(np.argmax(pre_res))
print(final_res)


# In[149]:


# result_txt = "result" + str(datetime.now()).split()[1] + ".txt"
print(len(final_res))
result_txt = "result001" + ".txt"
ids = 0
with open(result_txt, 'w') as out:
    out.write("id,category" + '\n')
    for value in final_res:
        out.write(str(ids) + "," + str(value) + '\n')
        ids += 1


# In[19]:


# summ = 0
# for s in test_texts[100:110]:
# #     summ += len(s)
#     print(s)
# # print(summ)

