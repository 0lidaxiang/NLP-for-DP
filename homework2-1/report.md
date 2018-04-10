# NLP Homework2 Report

過程所使用不同的參數對準確率所帶來的影響

validation_split:
#### 第１次
原始参数：
epochs = 100, validation_split = 0.2, batch_size = 10, verbose = 1
batch_size=10, verbose=0
0.70226
#### 2
epochs = 100, validation_split = 0.2, batch_size = 10, verbose = 1
batch_size=10, verbose=0
0.72168
#### 3
epochs = 100, validation_split = 0.5, batch_size = 10, verbose = 1
batch_size=10, verbose=0
0.69902
#### 4
epochs = 100, validation_split = 0.35, batch_size = 10, verbose = 1
batch_size=10, verbose=0
0.71521
#### 5
epochs = 100, validation_split = 0.425, batch_size = 10, verbose = 1
batch_size=10, verbose=0
0.68284
#### 6
epochs = 100, validation_split = 0.275, batch_size = 10, verbose = 1
batch_size=10, verbose=0
**0.72491**
#### 7
epochs = 100, validation_split = 0.33, batch_size = 10, verbose = 1
batch_size=10, verbose=0
0.68284

## batch_size
#### 1
epochs = 100, validation_split = 0.275, batch_size = 5, verbose = 1
batch_size=5, verbose=0
0.73139

#### 2
epochs = 100, validation_split = 0.275, batch_size = 2, verbose = 1
batch_size=2, verbose=0
0.68932

#### 3
epochs = 100, validation_split = 0.275, batch_size = 4, verbose = 1
batch_size=4, verbose=0
0.68284

#### 4
epochs = 500, validation_split = 0.275, batch_size = 5, verbose = 1
batch_size=5, verbose=0
0.68932

#### 5
epochs = 500, validation_split = 0.2, batch_size = 2, verbose = 1
batch_size=2, verbose=0
0.68608

#### 6
epochs = 500, validation_split = 0.05, batch_size = 2, verbose = 1
batch_size=2, verbose=0
0.71844

#### 7
epochs = 500, validation_split = 0.05, batch_size = 2, verbose = 1
batch_size=2, verbose=0
if value[0] > 0.7:
0.72815

#### 8
epochs = 500, validation_split = 0.05, batch_size = 2, verbose = 1
batch_size=2, verbose=0
if value[0] > 0.4:
0.67637

#### 9
model.fit(x = train_result, y = train_label, epochs = 200, validation_split = 0.1, batch_size = 4, verbose = 1)
scores = model.evaluate(x = train_result, y = train_label, batch_size=2)
print(scores)
res = model.predict(test_feature, batch_size=1, verbose=0)
 0.72168




 model.fit(x = train_result, y = train_label, epochs = 200, validation_split = 0.275, batch_size = 5, verbose = 1)
 scores = model.evaluate(x = train_result, y = train_label, batch_size=5)
 print(scores)
 res = model.predict(test_feature, batch_size=5, verbose=0)
 'male': 50











## 第2次
epochs = 100, validation_split = 0.2, batch_size = 2, verbose = 1
batch_size=2, verbose=0
0.70550

## 3
epochs = 200, validation_split = 0.2, batch_size = 2, verbose = 1
batch_size=5, verbose=0
0.70550

epochs = 200, validation_split = 0.1, batch_size = 2, verbose = 1
batch_size=2, verbose=0
0.70550
