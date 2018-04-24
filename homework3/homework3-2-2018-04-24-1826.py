
# coding: utf-8

# In[43]:


import warnings
warnings.filterwarnings('ignore')


# In[44]:


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


# In[45]:


# get texts data
category2idx = {'Japan_Travel': 0, 'KR_ENTERTAIN': 1, 'Makeup': 2, 'Tech_Job':  3, 'WomenTalk': 4,
                  'babymother': 5, 'e-shopping': 6, 'graduate': 7, 'joke': 8, 'movie': 9}

train_df_sample = pd.read_pickle('train.pkl').sample(frac=1, random_state=123)
# train_df_sample = pd.concat([train_pickle_df.text]).sample(frac=1)


# In[46]:


train_texts = train_df_sample.values
label_list = to_categorical(train_df_sample.category)

test_pickle_df = pd.read_pickle('test.pkl')
test_texts = test_pickle_df["text"].values


# In[47]:


train_texts_list = []
for text in train_texts:
    train_texts_list.append(text[0])
# print(len(train_texts_list), train_texts_list[0:10])
train_texts = train_texts_list


# In[48]:


# get word embedding vector
answer = word2vec.Word2Vec.load("word2vec1.model")
# print(type(answer))
word_vectors = answer.wv
wvv = word_vectors.vocab
wvv_keys = wvv.keys()
wvv_keys_list = list(wvv_keys)
# print(wvv_keys_list[:10]) #['櫻花林', '好比', '考科', '床上', '一點現', '記住', '寶寶的', '柔嫩', '不規則', '朴智妍']


# In[49]:


tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_texts)
sequences = tokenizer.texts_to_sequences(train_texts)
# max_doc_word_length = max(len(l) for l in train_texts)
max_doc_word_length = 200
sequences1 = pad_sequences(sequences, maxlen=max_doc_word_length, padding='post')
word_index = tokenizer.word_index
print("Found %s unique tokens" % len(word_index))
# data = pad_sequences(sequences)
# print("Shape of data tensor:" , data.shape)


# In[50]:


vocab_size = len(word_index) + 1
# create a weight matrix for words in training docs
embedding_matrix = np.zeros((vocab_size, 250))
for word, i in word_index.items():
    if word in wvv_keys_list:
        embedding_vector = answer[word]
#         if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
    else:
        embedding_matrix[i] = np.zeros((1, 250))


# In[51]:


del answer


# In[52]:


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


# In[ ]:


history = model.fit(x = sequences1, y = label_list, 
                    validation_split=0.1, 
                    batch_size= 3000,
                    epochs = 100, verbose = 1)


# In[ ]:


# loss_func_name = 'categorical_crossentropy'
# for value in history.history["categorical_crossentropy"]:
#     print(history)
model.save('my_model.h5') 


# In[ ]:


# evaluate the model
loss_accuracy = model.evaluate(sequences1[0:100], label_list[0:100], verbose=1)
print(type(loss_accuracy), loss_accuracy)


# In[ ]:


test_tokenizer = Tokenizer()
test_tokenizer.fit_on_texts(test_texts)
test_sequences = test_tokenizer.texts_to_sequences(test_texts)
test_sequences1 = pad_sequences(test_sequences, maxlen=max_doc_word_length, padding='post')


# In[ ]:


predict_res = model.predict(test_sequences1, batch_size= 32, verbose=0)
# print(len(predict_res), predict_res)

final_res = []
for pre_res in predict_res:
    final_res.append(np.argmax(pre_res))
print(final_res)


# In[ ]:


# result_txt = "result" + str(datetime.now()).split()[1] + ".txt"
print(len(final_res))
result_txt = "result001" + ".txt"
ids = 0
with open(result_txt, 'w') as out:
    out.write("id,category" + '\n')
    for value in final_res:
        out.write(str(ids) + "," + str(value) + '\n')
        ids += 1

