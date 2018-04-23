
# coding: utf-8

# In[6]:


import warnings
warnings.filterwarnings('ignore')


# In[7]:


#!/usr/bin/python3
# -*- coding: UTF-8 -*-
import time
from keras import metrics
import os
import numpy as np
import keras
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, SimpleRNN,Activation
from keras.layers.embeddings import Embedding
from datetime import datetime
from gensim.models import word2vec
import pandas as pd


# In[8]:


# get texts data
TRAINING_PATH = './data/training/'
TESTING_PATH = './data/testing/'

categories = [dirname for dirname in os.listdir(TRAINING_PATH) if dirname[-4:] != '_cut']
# print(len(categories), str(categories))

category2idx = {'Japan_Travel': 0, 'KR_ENTERTAIN': 1, 'Makeup': 2, 'Tech_Job':  3, 'WomenTalk': 4,
                  'babymother': 5, 'e-shopping': 6, 'graduate': 7, 'joke': 8, 'movie': 9}

train_pickle_df = pd.read_pickle('train.pkl')
train_texts = train_pickle_df["text"].values
train_labels = train_pickle_df["category"]
# print(len(train_texts), train_texts[0], train_labels[0])   

test_pickle_df = pd.read_pickle('test.pkl')
test_texts = test_pickle_df["text"].values
# print(len(test_texts), test_texts[0]) 


# In[9]:


# # process some data
train_labels_list = list(train_labels)
# print(type(train_labels_list), len(train_labels_list), train_labels_list[0])
embedding_matrix_len = len(train_labels_list)

label_id = 0
label_list = np.zeros((embedding_matrix_len, 10))
for label_val in train_labels_list:
    index = label_val - 1
    label_list[label_id][index] = 1
    label_id += 1
# print(len(label_list), label_list[0])


# In[10]:


# get word embedding vector
answer = word2vec.Word2Vec.load("word2vec1.model")
# print(type(answer))
word_vectors = answer.wv
wvv = word_vectors.vocab
wvv_keys = wvv.keys()
wvv_keys_list = list(wvv_keys)
# print(wvv_keys_list[:10]) #['櫻花林', '好比', '考科', '床上', '一點現', '記住', '寶寶的', '柔嫩', '不規則', '朴智妍']
# wvv_new_dict = dict(enumerate(wvv_keys_list))
# # word_index = len(wvv_keys_list)
# print(wvv_new_dict)


# In[11]:


tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_texts)
sequences = tokenizer.texts_to_sequences(train_texts)
max_doc_word_length = max(len(l) for l in train_texts)
sequences1 = pad_sequences(sequences, maxlen=max_doc_word_length, padding='post')
word_index = tokenizer.word_index
print("Found %s unique tokens" % len(word_index))
# data = pad_sequences(sequences)
# print("Shape of data tensor:" , data.shape)


# In[12]:


# print(len(sequences))
print(len(sequences[0]), sequences[0])
# print(len(sequences1[0]))
# print(wvv) #'同上': <gensim.models.keyedvectors.Vocab object at 0x7fd385b03d68>, 
# print(type(wvv)) <class 'dict'>


# In[13]:


vocab_size = len(word_index) + 1
# create a weight matrix for words in training docs
embedding_matrix = np.zeros((vocab_size, 250))
for word, i in word_index.items():
    if word in wvv_keys_list:
        embedding_vector = answer[word]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
# del answer


# In[31]:


# input_dim = 400
# output_dim = 100
embedding_layer = Embedding(vocab_size, 250, 
                            weights=[embedding_matrix], 
                            input_length= max_doc_word_length, 
                            trainable=False)
model = Sequential()
model.add(embedding_layer)

model.add(SimpleRNN(input_dim = 250, output_dim = 50, unroll=True))
nb_classes = 10
model.add(Dense( nb_classes, input_dim = 3971))
model.add(Activation('softmax'))
# early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=0, verbose=0, mode='auto')
model.compile(optimizer='adam',loss='categorical_crossentropy',  metrics=["acc"])
model.summary()


# In[32]:


history = model.fit(x = sequences1, y = label_list, 
                    validation_split=0.1, 
                    batch_size=32,
                    epochs = 1, verbose = 1)


# In[30]:


# loss_func_name = 'categorical_crossentropy'
# for value in history.history["categorical_crossentropy"]:
#     print(history)


# In[33]:


# evaluate the model
loss_accuracy = model.evaluate(sequences1[0:100], label_list[0:100], verbose=1)
print(type(loss_accuracy), loss_accuracy)


# In[34]:


test_tokenizer = Tokenizer()
test_tokenizer.fit_on_texts(test_texts)
test_sequences = test_tokenizer.texts_to_sequences(test_texts)
test_sequences1 = pad_sequences(test_sequences, maxlen=max_doc_word_length, padding='post')


# In[35]:


predict_res = model.predict(test_sequences1, batch_size= 32, verbose=0)
print(len(predict_res), predict_res)

final_res = []
for pre_res in predict_res:
    final_res.append(np.argmax(pre_res))
print(final_res)


# In[ ]:


# start_time = time.time()
# start = 0

# for embed_item_index in range(0, 487):
# #     train_result= np.asarray([embedding_matrix])
# #     train_label =  np.asarray([label_list[label_id]])
    
#     end = start + 100
#     train_result= articles_matrix[start:end]
#     train_label = label_list[start:end]
#     history = model.fit(x = train_result, y = train_label, validation_split=0.1, callbacks=[early_stopping], epochs = 5, verbose = 0)
#     model.save('my_model.h5')
#     scores = model.evaluate(x = train_result, y = train_label, batch_size=32, verbose = 0)
#     if  embed_item_index % 10 == 0 :  
#         end_time = time.time()
#         excution_time = end_time- start_time
#         excution_time_min = round(excution_time / 60, 2)
#         print(embed_item_index, " , loss : ", round(scores[0], 6), "   acc : " , scores[1],
#               "  minutes: " , excution_time_min)
# #     loss_func_name = 'categorical_crossentropy'
# # #     for value in history.history[loss_func_name]:
# #     print(history)
        
#     if embed_item_index < 40 :  
#         pass
        
#     else:
#         break
#     start += 100


# In[ ]:


# # result_txt = "result" + str(datetime.now()).split()[1] + ".txt"
# print(len(final_res))
# result_txt = "result001" + ".txt"
# ids = 0
# with open(result_txt, 'a') as out:
#     out.write("id,category" + '\n')
#     for value in final_res:
#         out.write(str(ids) + "," + str(value) + '\n')
#         ids += 1


# In[ ]:


# del model
# model = load_model('my_model.h5')

