# # Homework 2 of NLP for Deeplearning

#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import os
import math
import numpy as np
import pandas as pd

from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Activation,Dense, Dropout
from datetime import datetime,timedelta

TRAIN_FILE_PATH='./data/training_data(1000).xlsx'
TEST_FILE_PATH='./data/testing_data.xlsx'

train_df = pd.read_excel(TRAIN_FILE_PATH)
test_df = pd.read_excel(TEST_FILE_PATH)

cols = ['survived', 'pclass', 'sex','name', 'boat', 'age', 'sibsp', 'parch', 'fare', 'embarked']
train_df = train_df[cols]
test_df = test_df[cols]


def PreprocessData(raw_df):
    #Remove the 'name' col 雙親或子女在船上的數量、 Ticket
    df = raw_df.drop(['name'], axis=1)
    age_mean = df['age'].mean()
    df['age'] = df['age'].fillna(age_mean)
    fare_mean = df['fare'].mean()
    df['fare'] = df['fare'].fillna(fare_mean)

    sss = df['boat']


    ids = 0
    for vs in sss:
        if type(vs) == str:
            df.loc[ids, "boat"] = 24
        else:
            if math.isnan(float(vs)):
                df.loc[ids, "boat"] = 0
            else:
                df.loc[ids, "boat"] = 23
        ids+=1

    df['sex'] = df['sex'].map({'female': 0, 'male': 1}).astype(int)

    x_OneHot_df = pd.get_dummies(data=df, columns = ["embarked"])
    ndarray = x_OneHot_df.values

    label = ndarray[:,0] #answer('survived' col)
    Features = ndarray[:,1:] #input(other cols)

    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))

    scaledFeatures = minmax_scale.fit_transform(Features)

    return scaledFeatures, label



boat_val = test_df["boat"]
train_result, train_label = PreprocessData(train_df)
test_feature, test_label = PreprocessData(test_df)


model = Sequential()
model.add(Dense(units=80, input_dim=10, kernel_initializer='uniform'))
model.add(Activation('relu'))

model.add(Dense(units=60, kernel_initializer='uniform'))
model.add(Activation('relu'))

# model.add(Dense(units=30, kernel_initializer='uniform'))
# model.add(Activation('relu'))

model.add(Dense(units=1, kernel_initializer='uniform'))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.summary()


model.fit(x = train_result, y = train_label, epochs = 500, validation_split = 0.1, batch_size = 5, verbose = 1)
scores = model.evaluate(x = train_result, y = train_label, batch_size=5)
res = model.predict(test_feature, batch_size=5, verbose=0)

print(scores)

result_txt = "result" + str(datetime.now()).split()[1] + ".txt"
ids = 0
with open(result_txt, 'a') as out:
    out.write("id,survived" + '\n')
    for value in res:
        if value[0] > 0.6:
            out.write(str(ids) + "," + str(1) + '\n')
        else:
            out.write(str(ids) + "," + str(0) + '\n')
        ids += 1
