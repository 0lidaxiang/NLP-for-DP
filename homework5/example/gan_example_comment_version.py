#各位同學好 我是NLP助教 羅上堡 M10615012
#這是一份重頭到尾很多註解的Code
#利用Keras去實現的
#這份Code是我親自寫的 如果覺得排版有嚴重問題 請見諒QQ..
#然而這份Code不能直接執行
#因為這次的作業如果以第一次做GAN來說 偏難 所以註解會多一點
#如果嫌註解很煩 請見諒 不然第一次做GAN會遇到很多問題的 (PS:雖然不覺得打很多註解會有效幫助各位同學實現就是了...)

#這份Example Code是以Pix2Pix GAN的架構去寫的 ，類似關鍵字可以查Conditional GAN
#其中這份Code有<your creativity>,<your answer>,<your choose>等字眼 就是自己填答案
#如果同學想用很簡單的方法去實現 可以只參考這篇的寫法即可==>尤其是訓練/定義網路的方式

###############################################################
#Import package or lib
###############################################################
import datetime
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#如果以上沒import成功 代表沒安裝過 利用網路資源或是pip install 安裝巴
import sys
import os
#對於很多有用CUDA進行加速的套件來說 這種方法 比較好控制你是要用CPU還GPU ==> tensorflow可以獨立配置 先不考慮
# -1 ==> Use CPU
# 0 or 1 or 0,1 ==> Use GPU
#但是大部分都是用CPU 所以這邊直接打-1當範例Code
os.environ["CUDA_VISIBLE_DEVICES"]='-1' 
#如果有同學import下面這行 有問題 那就是我們python的版本不同 或是scipy版本不同 可以自行去找類似的imread其他的都可
#也有可能你的是 from scipy.misc import imread,imresize,imsave 
#助教的版本
#Python 3.5.2
#Scipy 1.0.0
from scipy.misc.pilutil import imread,imresize

###############################################################
#Import keras 所需要的東西
###############################################################
#這邊的*你可以 後面在改 因為沒人會知道 你到底要用甚麼方式去實現 以及要用那些層
from keras.models import *
from keras.layers import *
from keras.layers.merge import *
from keras.optimizers import *


###############################################################
# Training Data List Creat
###############################################################
train_real_data_dir = r'.\NLP_data\Training\Real\*'
train_white_data_dir = r'.\NLP_data\Training\White\*'

real_list = glob.glob(train_real_data_dir)
train_real_data_list = []
train_real_data_list.extend(real_list)

white_list = glob.glob(train_white_data_dir)
train_white_data_list = []
train_white_data_list.extend(white_list)

###############################################################
# Define D and G and parameter
###############################################################
img_row = img_col = 128 #先規定成128*128 不過由於Pixel2Pixel架構的關係 最好是2的次方數 (2^n)
channels = 1    # 1 is gray
                # 3 is RGB
img_shape=(channels,img_row,img_col) # or (img_row,img_col,channels)
#2018.05.03補充==>這邊Img_shape在Keras中 預設是『img_row,img_col,channels』 ==>Channels是放在後面的
#                 所以這邊看你們怎麼設定img_shape
#                 如果你用我這種寫法『channels,img_row,img_col』 你後面在Call Conv2D等等的東西 需要加入『data_format='channels_first'』
#                 不過我記得有可以在剛開始一次設定是channels_first還是channels_last，就請各位去查查囉。
#
def dis(input_shape):
    #這部分是Discriminator 判別器的定義
    #記住我們是Conditional GAN所以有兩個輸入，然而我們的Conditional Input 就是圖
    #所以兩個Input都是輸入影像
    #Real Pair ==> (真實圖，抹白圖)
    #Fake Pair ==> (仿照圖，抹白圖)
    #輸出是Patch GAN 都經過Conv2D來提取
    #輸出一張n*n的Label圖==>沒錯，不像一般是一維的Label。
    def conv_block(input):
        #<your creativity>
        #你想輸入別的參數都可以，請隨意修改==>像是可以指定filter、Stride的大小之類的 
        #你不想用def的方式定義網路也是可以的(反正Example Code這種寫法 只是為了比較好看)
        #通常是一個Conv2D + Activation function (ReLU或是LeakyReLU等等)
        #然後配一個BatchNormalization(option)
        return x
     
    img_A = Input(input_shape)      #真實答案 or 虛假答案 
    img_B = Input(input_shape)      #助教塗白的真實圖
    combined_img = Concatenate(axis=1)([img_A,img_B])
    # 通常是一連串 你想要幾個都可以 然後取的Filter size 會一直增倍下去
    x = conv_block(<your creativity>)
    x = conv_block(<your creativity>)
    x = conv_block(<your creativity>)
    x = conv_block(<your creativity>)
    # 輸出 通常是1維輸出 不過這裡的Example Code 是Patch GAN的Dis 所以你要直接Dense or Conv2D都是可以的
    # Patch gan是說 只要
    # Select 1 Patch-GAN Dis 
    x = Conv2D(<your creativity>)(x)
    # Select 2 Dense 
    # 不過做Desne前 應該需要Flatten喔 不過我有點懶 就不提供flatten的代碼了
    x = Dense(<your creativity>)(x)
    
    model = Model([img_A,img_B], x)
    print('Model_Discriminator:')
    model.summary()
    return model
    
def gen(input_shape,** kwargs):
    #這部分是Generator 生成器的定義
    #其實就是FCN(全卷積網路)
    #輸入是抹白圖 輸出是仿照真實圖
    #就很像是圖像的AutoEncoder
    #然後Pix2Pix的Generator是U-NET 『也是FCN的一種架構』
    #在Decoder的時候 會將Encoder的同Size輸出 串接進來一起Decoder
    #如果有詳細問題 可以去看U-NET的架構
    #有個重點就是 記住要讓Input==>Encoder==>Decoder==>Output『輸入輸出』維度一樣。
    #2015.05.04裡面的架構你可以隨便修改，但為了符合 輸入抹白圖 輸出仿照真實圖 所以才建議FCN架構。
    def conv_block(input):
        #<your creativity>
        #別擔心重複def問題 因為是在gen裡面才def 所以不會有重複宣告問題 讚嘆python
        #通常一樣Conv2D + Activation function
        #然後配一個BatchNormalization(option)
        return x
    
    def deconv_block(input,skip_input):
        #<your creativity>
        #通常作法UpSampling2D + Conv2D + activation function 
        #然後配一個BatchNormalization(option)
        x = Concatenate(axis=???)([<your creativity>,skip_input])
        #首先這邊Concatenate 有axis這個參數 因為要做串接前一層的輸入(skip_input)去調整一下axis 不然網路的輸出會跟你想的不同
        #不過如果你連Concat都沒辦法 就是你Size根本不對或是觀念不夠喔
        return x

    img_A = Input(input_shape)  # size (None,1,128,128)
    #這邊很類似AutoEncoder
    #輸入一張圖==>降維==>升維==>輸出一張圖 ，並且輸入與輸出的圖大小一致

    # Encoder ==> 將圖做降維
    x1 = conv_block(<your creativity>) #降維step1   # output_size (None,filter_A,A,A)
    x2 = conv_block(<your creativity>) #降維step2   # output_size (None,filter_B,B,B)  #當然A>B>C>D 
    x3 = conv_block(<your creativity>) #降維step3   # output_size (None,filter_C,C,C)  #就是輸出的Size愈來愈小
    x4 = conv_block(<your creativity>) #降維step4   # output_size (None,filter_D,D,D)  #然而filte_A<=filter_B<=filter_C<=filter_D
                                                                                         #通常是f_D = 2*f_C = 4*f_B = 8*f_A 就是愈取愈多張feature map(filter參數)
                                                                                         #這邊如果你用非常多層 怕CPU的夥伴會很難過(跑不動) 所以Example是給4層而已
                                                                                         #如果你覺得feature map取太多很慢 你是可以把他調很小的 例如1張之類的...
    # Middle_vector
    x5 = conv_block(<your creativity>) #重新提取同維度的特徵  #size (None,filter_D,D,D)

    # Decoder ==> 將降維後的Middle Vector 升維至原圖

    d1 = deconv_block(<your creativity>) #升維Step1 # output_size (None,filter_D',D,D)
    d2 = deconv_block(<your creativity>) #升維Step2 # output_size (None,filter_C',C,C)
    d3 = deconv_block(<your creativity>) #升維Step3 # output_size (None,filter_B',B,B)
    d4 = deconv_block(<your creativity>) #升維Step4 # output_size (None,filter_A',A,A) #沒錯就是反操作 假設你conv 是每次經過 降兩倍的size 那deconv就是升兩倍的size
                                                    # 你會發現filter_D多了個'這是因為你要將skip_input做Concat 所以Size不變 他會Concat再filter_D那個維度
                                                    # Example A=(50,128,128) B=(50,128,128) = (filter_num,w,h)
                                                    # output_size =(100,128,128)
                                                    # 而不是 (50,128,256) #你如果變成這樣 你會發現會一直爆維度問題
                                                    # U-NET會有Skip-input所以你的deconv_block需要多一個輸入 可以參考投影片
    
    d5 = UpSampling2D(<your creativity>)(d4)
    
    out_img = Conv2D(<your creativity>,activation=<your choice>)(d5) #這邊記住 是output_img 所以你的fiter要調整成 1因為我們這次的作業是灰階圖 不是彩色的
                                                              #這邊activation建議用tanh or sigmoid
                                                              #提醒：
                                                              #我們的Data處理的時候 會將值域從0~255調整到0~1 所以輸出的激活函數 建議用tanh or sigmoid
    
    model = Model(img_A, out_img)
    print('Model_Generator:')
    model.summary()
    return model

###############################################################
#就規定好shape並且傳給G跟D來創構Modle
###############################################################

input_shape=(channels,img_row,img_col) # or (img_row,img_col,channels)
crop_shape=(img_row,img_col)
G = gen(input_shape)
D = dis(input_shape)


###############################################################
# 定義訓練D的模型
###############################################################
#定義你的優化器巴 SGD Adam Adamax Adadelta rmsprop etc... 都可以 隨便
#只是學習率通常別調太高 ...(ps:容易壞掉QQ)   我自己是用0.0002 Adam
D_optimizer = <your creativity>
#如果你是用Patch GAN conv2D做結尾的 建議用MSE
#如果你是用Dense直接變成1維 則沒限制
D.compile(loss=<your creativity>, optimizer=D_optimizer,metrics=['accuracy'])
#Loss 跟投影片一樣 MSE 或 Cross Entropy 隨便 都可以試
D.summary()

###############################################################
# 定義訓練G的模型
###############################################################
#沒錯上課說過
#再訓練G的時候 需要把D引入近來 並且Fix它 寫法就在下面
#當然我們還是先定義優化器 ==>記住 通常是跟D的優化器設定一樣 不過沒人說這樣一定好 隨便你的選擇
#這邊我們將這種網路叫做AM (對抗模型) 
AM_optimizer = <your creativity>
img_A = Input(input_shape)          #真實圖 
img_B = Input(input_shape)          #助教塗白的真實圖
fake_A = G(img_B)                   #先將塗白的塗去預測出fake image ==>期望生成真實圖
D.trainable=False                   #讓D在AM模型Fix
valid = D([fake_A,img_B])           #將fake img 跟抹白的圖給D去判別 得出判決結果valid
AM = Model([img_A,img_B],[valid,fake_A]) # 定義model 我們這邊的輸出 有包含D的結果以及 
                                         # G收到抹白的圖所產生的fake圖
AM.compile(loss=[<your creativity>,<your creativity>],loss_weights=[1,1],optimizer=optimizer)
                                         # 這邊的Loss通常是MSE或MAE,其他也可以
                                         # 第一個Loss 是 Dis的輸出Loss
                                         # 第二個Loss 你就想像是AutoEncoder的訓練方式 輸入抹白 輸出仿照圖  跟真實圖做Loss
                                         # loss_weights是為了加快訓練效果 如果不想調整 都為1即可 通常是AutoEncoder的Loss weight調大
AM.summary()

###############################################################
# Define Image Generator
###############################################################
#這裡呢 就是產生Image 給你的Model的地方
#我懶得用Fit_generate 就每個epoch 呼叫一次巴OWO
def generator_training_Img(real_list_dir,white_list_dir,resize=None,batch_size=32):
    batch_real_img=[]
    batch_white_img=[]
    for _ in range(batch_size):
        real_img = imread(np.random.choice(real_list_dir),mode='L')
        white_img = imread(np.random.choice(white_list_dir),mode='L')
        if resize:
            real_img = imresize(real_img,resize)
            white_img = imresize(white_img,resize)
        batch_real_img.append(real_img)
        batch_white_img.append(white_img)
    batch_real_img = np.array(batch_real_img)/127.5-1
    batch_real_img = np.expand_dims(batch_real_img,axis=1)
    batch_white_img = np.array(batch_white_img)/127.5-1
    batch_white_img = np.expand_dims(batch_white_img,axis=1)
    return batch_real_img,batch_white_img


###############################################################
# Training Phase
###############################################################
#Epoch給個小基準 在我自己的實驗是這樣的
#batch ==> 16 all_epoch ==> 12000
#batch ==> 32 all_epoch ==> 7000
#batch ==> 64 all_epoch ==> 5600
#你要Random我也是沒差拉 

batch_size=<your creativity>
all_epoch=<your creativity>
#這邊是在定義答案 valid(Real)是1 fake是0 
#size當然是『D的輸出Size』囉
#2018.05.04補充：如果你是直接Dense成1維輸出的大小，就只要定義(batch_size,1)就可以了
#因為註解寫了答案 所以填入你的答案巴 
#別懷疑 是四維的 你如果要改成三維 上面的網路架構不會符合這種輸入 你可能需要對網路架構Reshape之類的動作
valid = np.<your answer>s((batch_size,<your answer>,<your answer>,<your answer>))
fake  = np.<your answer>s((batch_size,<your answer>,<your answer>,<your answer>))

#純粹紀錄時間    
start_time=datetime.datetime.now()


for now_iter in range(all_epoch):
    #呼叫generator_Img 輸出真實圖與抹白圖
    ori_img,white_img = generator_training_Img(real_list_dir=train_real_data_list,
                                               white_list_dir=train_white_data_list,
                                               resize=(img_row,img_col),
                                               batch_size=batch_size)
    ###################################
    #Training Discriminator Phase
    ###################################
    fake_A = G.predict(<your answer>) # 先讓抹白圖給Generator去預測Fake圖 
    
    #接下來有兩個Loss 分別是Real loss & Fake loss
    #一組是輸入 真實圖與抹白圖 答案給 1 因為這是Real Pair
    #一組是輸入 虛假圖與抹白圖 答案給 0 因為這是Fake Pair
    #對於D來說 它是要『正確分類』 所以他要學習 Fake Pair是假的 Real Pair是真的
    
    #記得去查Train_on_batch的API喔 
    D_loss_Real = D.train_on_batch([<your answer>,<your answer>],<your answer>)
    D_loss_Fake = D.train_on_batch([<your answer>,<your answer>],<your answer>)
    D_loss = 0.5 * np.add(D_loss_Real,D_loss_Fake)
    
    ###################################
    #Training Generator Phase
    ###################################
    #這邊是訓練Generator
    #我們知道這邊是要讓D認為這是Real Pair (這樣才有對抗==> 訓練D時,告訴D     Fake Pair是假的
    #                                                      訓練G時,要讓D認為 Fake Pair是真的) 
    #輸入 真實圖與抹白圖 答案是 1還是0? 你要讓D誤會 所以是?
    #另外一個答案是真實圖 
    #如果不清楚要填啥 可以去看上面定義AM model時，你希望這個輸出愈像甚麼愈好?
    #                                             (PS:你希望假的圖愈像什麼愈好?)
    G_loss = AM.train_on_batch([<your answer>,<your answer>],[<your answer>,<your answer>])
    

    end_time = datetime.datetime.now() - start_time
    #這邊就是單純顯示訓練過程的結果
    #會print出D loss,acc 以及G的兩個loss並且每個epoch跑的時間多少
    print("[Epoch %d/%d] [D loss: %f, acc: %3d%%] [G loss1: %f,loss2: %f] [time:%s]" % (now_iter,all_epoch,D_loss[0],D_loss[1]*100,G_loss[0],G_loss[1],end_time))



###############################################################
# Display 你的結果 如果沒辦法顯示 就去查plt figure savefig API 你就可以把它存成圖出來給自己看學習狀況
# 當然你覽的話 直接刪掉也是可以的 只是你就只能用一值跳動的loss觀察是不是有在學習
# 顯示方式 【預測圖 真實圖 抹白圖】*n次
###############################################################

plt.gray()
n = 2
r,c=(n,3)
plt.figure(figsize=(c*6,r*6))
for i in range(r):
    ori_img,white_img = generator_Img(real_list_dir=train_real_data_list,
                                      white_list_dir=train_white_data_list,
                                      resize=(img_row,img_col),
                                      batch_size=batch_size)
    ax = plt.subplot(r, c, i*c + 1)
    a = G.predict(white_img).reshape(img_row,img_col)
    plt.imshow(a)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax = plt.subplot(r, c, i*c + 2)
    a = ori_img.reshape(img_row,img_col)
    plt.imshow(a)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax = plt.subplot(r, c, i*c + 3)
    a = white_img.reshape(img_row,img_col)
    plt.imshow(a)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)   
plt.show()

#####################################
#假設你需要輸出Image 去CSV檔案
#####################################

# step 1 讀取Test Data

# step 2 一樣用G.Predict 

# step 3 將Image 串接起來 並 儲存成CSV 

# 自己寫巴 因為其實不難 在看得懂Training data 讀取的話
