
# coding: utf-8

# In[11]:



import warnings
warnings.filterwarnings('ignore')
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


# In[12]:


#idenity paramaters
word_vec_len = 256


# In[13]:


# get data
# get texts data
category2idx = {'AllTogether': 0, 'Baseball': 1, 'Boy-Girl': 2, 'C_chat':  3, 'CVS': 4,
                  'GameSale': 5, 'GetMarry': 6, 'Lifeismoney': 7, 'LoL': 8, 'MH': 9, 'MLB': 10, 'Mobilecomm': 11, 
                'movie': 12,'MuscleBeach':  13, 'NBA': 14,  'SENIORHIGH': 15, 'Stock': 16, 
                'Tennis': 17, 'Tos': 18, 'WomenTalk': 19}

train_df_sample = pd.read_pickle('train.pkl').sample(frac=1, random_state=123)


# In[14]:


# print(type(train_df_sample), len(train_df_sample))
# print(train_df_sample.label)


# In[15]:


train_texts = train_df_sample.values
label_list = train_df_sample.label

test_pickle_df = pd.read_pickle('test.pkl')
test_texts = test_pickle_df["text"].values
# print(len(test_texts), test_texts[0]) 


# In[16]:


train_texts_list = []
for text in train_texts:
    train_texts_list.append(text[0])
# print(len(train_texts_list), train_texts_list[0])


# In[17]:


# get word embedding vector
answer = word2vec.Word2Vec.load("word2vec_20180430.model")
word_vectors = answer.wv
wvv = word_vectors.vocab
wvv_keys = wvv.keys()
wvv_keys_list = list(wvv_keys)


# In[18]:


print(len(wvv), len(wvv_keys_list))


# In[19]:


texts_list = []
for text in train_texts_list:
#     print(text)
    texts_list.append(text)
    
for text in test_texts:
    texts_list.append(text)


# In[20]:


print(type(texts_list), len(texts_list))
# print(label_list[0:10])


# In[21]:


del word_vectors, wvv, train_texts_list
# print(texts_list[0], "\n")
# print(texts_list[1], "\n")
# print(texts_list[2], "\n")


# In[22]:


get_ipython().run_cell_magic(u'time', u'', u'tokenizer1 = Tokenizer(num_words = 20000)\ntokenizer1.fit_on_texts(texts_list)\nsequences = tokenizer1.texts_to_sequences(texts_list)  # word index list\nmax_doc_word_length = 200\nsequences1 = pad_sequences(sequences, maxlen=max_doc_word_length, padding=\'post\')\n\nprint("Found %s unique tokens" % len(tokenizer1.word_index))')


# In[23]:


get_ipython().run_cell_magic(u'time', u'', u'word_index = tokenizer1.word_index\nword_counts = tokenizer1.word_counts\nprint(len(word_index), word_index["\u5de7\u5408"], len(word_counts))\nnum_words = max(max(sequences))\nprint(num_words)\n\nword_index_keys_list = list(word_index.keys())\nword_index_values_list = list(word_index.values())\n    \n\ndocs_index = tokenizer1.texts_to_sequences(texts_list)\nprint(num_words)\n\n\n# real_word_index = np.zeros((1, num_words))\n# real_word_index = dict()\n# print(type(real_word_index))\n\n# ss = 0\n# for doc_index in docs_index:\n#     print(ss)\n#     for words_index in doc_index:\n#         key_v = word_index_keys_list[word_index_values_list.index(words_index)]\n# #         print(words_index, word_index_keys_list[word_index_values_list.index(words_index)])\n# #         if  key_v in real_word_index.keys():\n#         if word_counts[key_v] < 19992:\n#             real_word_index[key_v] = words_index\n#     ss += 1\n# print(real_word_index.shape)\n# print( max(docs_index) ) #\u5f97\u5230\u8bcd\u7d22\u5f15\n# print(max(sequences), sequences[1])')


# In[24]:


# sslist = list(word_index)
# print(sslist[19989:20000])


# In[25]:


# key_v = word_index_keys_list[word_index_values_list.index(19989)]
# print(key_v, word_counts[key_v])

# key_v = word_index_keys_list[word_index_values_list.index(19991)]
# print(key_v, word_counts[key_v])
# print(word_index.get("再練"))


# In[26]:


# print(type(tokenizer1.word_index) , len(tokenizer1.word_index)) 
# print(len(sequences1), tokenizer1.num_words) 
# # print(tokenizer1.word_index)
# ss = 0
# for k in tokenizer1.word_docs:
#     if tokenizer1.word_docs[k] > 100:
#         print(k, tokenizer1.word_docs[k])
#         ss += 1
# print(ss)


# In[27]:


# ss = 0
# for k in tokenizer1.word_docs:
#     if tokenizer1.word_docs[k] < 3 :
# #         print(k, tokenizer1.word_docs[k])
#         del word_index[k]
#         ss += 1
# print(ss)


# In[28]:


get_ipython().run_cell_magic(u'time', u'', u'# user 46min 33s,\nvocab_size = 20000 + 1\n# vocab_size = len(word_index) + 1\nprint(vocab_size, word_vec_len)\n# create a weight matrix for wordds in training docs\nembedding_matrix = np.zeros((vocab_size, word_vec_len))\nss = 0\n\nfor word, i in word_index.items():\n    if ss < 20000:\n        if i < 20000:\n            embedding_vector = answer[word]\n            embedding_matrix[i] = embedding_vector\n    else:\n        break\n#         embedding_matrix[i] = np.zeros((1, word_vec_len))   ')


# In[29]:


# del answer


# In[30]:


# # network model
print(embedding_matrix.shape, len(embedding_matrix[10]))
# (571339, 256) 256


# In[35]:


# image_width = 250
# image_height = 3
# input_channel = 1

embedding_layer = Embedding(
                            input_dim= embedding_matrix.shape[0],
                            output_dim= 256,
                            weights=[embedding_matrix], 
                            input_length = 200,
                            trainable=False)
# model = getModel(embedding_layer, image_width, image_height, input_channel)
model = Sequential()
model.add(embedding_layer )
# model.add(Reshape(4, 2))
# conv2D = 
# model.add(Conv1D(64, 3, 
#             border_mode='same',
#             input_shape=(3, 256)))
# model.output_shape == (None, 64, 32, 32)

# model.add(Flatten())
model.add(Conv1D(256, 3,
#                 filters=(9), 
#                 kernel_size = (256),
                padding = 'same', 
#                  input_shape=(9, 32, 32),
#                  input_shape=(10, 32)
                )
         )
# model.add(MaxPooling1D(pool_size=2))    
model.add(Flatten())
model.add(Dense(2, activation='relu'))
# early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=0, verbose=0, mode='auto')
model.compile(optimizer='sgd',loss='mae')
model.summary()


# In[36]:


# sequences2 = np.reshape(sequences1,(2, 20000, 200)) 
print(sequences1.shape, label_list.shape, label_list[0])

Y_label_list = np.zeros((36000, 2))
for ids in range(0, 36000):
    Y_label_list[ids][0] = label_list[ids][0]
    Y_label_list[ids][1] = label_list[ids][1]
print(Y_label_list.shape)


# In[37]:


# Y_label_list[:10]


# In[ ]:


history = model.fit(x = sequences1[0:36000], 
                    y = Y_label_list, 
#                     validation_split=0.1, 
                    batch_size= 18000,
                    epochs = 100, verbose = 1)


# In[ ]:


X_train =sequences1


# In[ ]:


# evaluate the model
loss_accuracy = model.evaluate(X_train[0:100], Y_label_list[0:100], verbose=1)
print(type(loss_accuracy), loss_accuracy)

test_sequences1 = X_train[36000:40000]

predict_res = model.predict(test_sequences1, batch_size= 3600, verbose=0)
# print(len(predict_res), predict_res)

final_res = []
for pre_res in predict_res:
    final_res.append(pre_res)
print(final_res)


# In[ ]:


# result_txt = "result" + str(datetime.now()).split()[1] + ".txt"
print(len(final_res))
result_txt = "server_result001" + ".txt"
ids = 0
with open(result_txt, 'w') as out:
    out.write("id,good,bad" + '\n')
    for value in final_res:
        out.write(str(ids) + "," + str(int(value[0])) + "," + str(int(value[1])) + '\n')
        ids += 1

