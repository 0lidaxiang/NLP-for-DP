
# coding: utf-8

# In[1]:


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
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D,Activation
from keras.layers.embeddings import Embedding
from datetime import datetime
from gensim.models import word2vec
from collections import *
import pandas as pd


# In[ ]:


# get data


# In[17]:


# network model

def getModel(image_width, image_height, input_channel):
    model = Sequential()
    conv2D = Conv2D(filters=9, kernel_size = (3,3),
                padding = 'same', 
                input_shape = (image_width, image_height, input_channel))
    model.add(conv2D)
    model.add(MaxPooling2D(pool_size=(2,2)))    
    model.add(Flatten() )
    model.add(Dense(10, activation='softmax'))
    return model


# In[18]:


image_width = 256
image_height = 256
input_channel = 3


model = getModel(image_width, image_height, input_channel)
# early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=0, verbose=0, mode='auto')
model.compile(optimizer='adam',loss='categorical_crossentropy',  metrics=["accuracy"])
model.summary()

