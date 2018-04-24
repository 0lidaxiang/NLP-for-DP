# NLP Homework 3

Model description (1%)
● How do you improve your performance (3%)
● Experiment settings and results (1%)
○ Ex. Epochs, training time, hyperparameters, etc.

## 作業3-1和處理資料
第一個作業基本就正常按照投影片做，後來根據kaggle上面同學分享的經驗改進了一下：用pkl存儲所有文本。

一開始思路比較混亂，甚至沒有考慮到要生成文章對應類別list，也就是 label_list。開始我是對每篇文章的每個切割後字詞的 word_embedding 加總取平均後去訓練，但是這樣預測出來結果都是一個分類，明顯錯了。後來請教助教的時候發現應該是壓縮取平均的過程中把每篇文章的差異性消除了，所以才會導致分成一個類別。


## 訓練
```
history = model.fit(x = sequences1, y = label_list, 
                    validation_split=0.1, 
                    batch_size=100,
                    epochs = 50, verbose = 1)
```
