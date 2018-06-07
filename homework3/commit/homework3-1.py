
# coding: utf-8

import pandas as pd
from gensim.models.word2vec import Word2Vec

train_df = pd.read_pickle("./train.pkl")
test_df = pd.read_pickle("./test.pkl")
train_df.head()

corpus = pd.concat([train_df.text, test_df.text]).sample(frac=1)
corpus.head()

model = Word2Vec(corpus, size=250, iter= 8, workers=3, min_count=5, negative=3, max_vocab_size=None, window=5)

model_file_name = 'word2vec_20180425.model'
model.save(model_file_name)
model = Word2Vec.load(model_file_name)


def most_similar(w2v_model, words, topn=10):
    similar_df = pd.DataFrame()
    for word in words:
        try:
            similar_words = pd.DataFrame(w2v_model.wv.most_similar(word, topn=topn), columns=[word, 'cos'])
            similar_df = pd.concat([similar_df, similar_words], axis=1)
        except:
            print(word, "not found in Word2Vec model!")
    return similar_df


# In[18]:


res = model.most_similar("日本", topn=100)

print("日本")
print("相似詞前 100 排序:")

for v in res:
    print(v[0], ",", v[1])

print('\r\n\n')

res1 = model.similarity('日本', '台灣')
print('日本', '台灣 : ', res1)


# In[17]:


# model = Word2Vec(corpus, size=250, iter= 8(不能擴到20), workers=3, min_count=5, negative=3, max_vocab_size=None, window=5)
# most_similar(model, ['懷孕', '網拍', '補習', '東京', 'XDD', '金宇彬','化妝品', '奧斯卡', '主管', '女孩'])


# In[11]:


# model = Word2Vec(corpus, size=250, iter= 10, workers=3, min_count=5, negative=3, max_vocab_size=None, window=5)
# most_similar(model, ['懷孕', '網拍', '補習', '東京', 'XDD', '金宇彬','化妝品', '奧斯卡', '主管', '女孩'])


# In[8]:


# model = Word2Vec(corpus, size=250, iter= 10, workers=3, min_count=5, negative=5, max_vocab_size=None, window=5)
# most_similar(model, ['懷孕', '網拍', '補習', '東京', 'XDD', '金宇彬','化妝品', '奧斯卡', '主管', '女孩'])


# In[ ]:


# %%time
# res = model.most_similar("日本", topn=100)

# print("日本")
# print("相似詞前 100 排序:")

# for v in res:
#     print(v[0], ",", v[1])

# print('\r\n\n')

# res1 = model.similarity('日本', '台灣')
# print('日本', '台灣 : ', res1)
