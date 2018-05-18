import datetime
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]='-1' 
from scipy.misc.pilutil import imread,imresize
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
img_row = img_col = 128 
channels = 1    
img_shape=(channels,img_row,img_col) 

def dis(input_shape):
    def conv_block(input):
        #<your creativity>
        return x
     
    img_A = Input(input_shape)      
    img_B = Input(input_shape)      
    combined_img = Concatenate(axis=1)([img_A,img_B])
    x = conv_block(<your creativity>)
    x = conv_block(<your creativity>)
    x = conv_block(<your creativity>)
    x = conv_block(<your creativity>)
   
    x = Conv2D(<your creativity>)(x) or   x=Dense(<your creativity>)(x)
    
    model = Model([img_A,img_B], x)
    print('Model_Discriminator:')
    model.summary()
    return model
    
def gen(input_shape,** kwargs):
    
    def conv_block(input):
        #<your creativity>
        return x
    
    def deconv_block(input,skip_input):
        #<your creativity>
        x = Concatenate(axis=???)([<your creativity>,skip_input])
        return x

    img_A = Input(input_shape)  


    x1 = conv_block(<your creativity>) 
    x2 = conv_block(<your creativity>) 
    x3 = conv_block(<your creativity>) 
    x4 = conv_block(<your creativity>) 

    x5 = conv_block(<your creativity>) 

    d1 = deconv_block(<your creativity>) 
    d2 = deconv_block(<your creativity>) 
    d3 = deconv_block(<your creativity>) 
    d4 = deconv_block(<your creativity>) 

    d5 = UpSampling2D(<your creativity>)(d4)
    
    out_img = Conv2D(<your creativity>,activation=<your choice>)(d5) 

    model = Model(img_A, out_img)
    print('Model_Generator:')
    model.summary()
    return model


input_shape=(channels,img_row,img_col) # or (img_row,img_col,channels)
crop_shape=(img_row,img_col)
G = gen(input_shape)
D = dis(input_shape)


D_optimizer = <your creativity>
D.compile(loss=<your creativity>, optimizer=D_optimizer,metrics=['accuracy'])
D.summary()

 
AM_optimizer = <your creativity>
img_A = Input(input_shape)          
img_B = Input(input_shape)         
fake_A = G(img_B)                   
D.trainable=False                   
valid = D([fake_A,img_B])           
AM = Model([img_A,img_B],[valid,fake_A])  
                                         
AM.compile(loss=[<your creativity>,<your creativity>],loss_weights=[1,1],optimizer=optimizer)
AM.summary()

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



batch_size=<your creativity>
all_epoch=<your creativity>

valid = np.<your answer>s((batch_size,<your answer>,<your answer>,<your answer>))
fake  = np.<your answer>s((batch_size,<your answer>,<your answer>,<your answer>))
    
start_time=datetime.datetime.now()
for now_iter in range(all_epoch):
    ori_img,white_img = generator_training_Img(real_list_dir=train_real_data_list,
                                               white_list_dir=train_white_data_list,
                                               resize=(img_row,img_col),
                                               batch_size=batch_size)
    ###################################
    #Training Discriminator Phase
    ###################################
    fake_A = G.predict(<your answer>) 

    D_loss_Real = D.train_on_batch([<your answer>,<your answer>],<your answer>)
    D_loss_Fake = D.train_on_batch([<your answer>,<your answer>],<your answer>)
    D_loss = 0.5 * np.add(D_loss_Real,D_loss_Fake)

    G_loss = AM.train_on_batch([<your answer>,<your answer>],[<your answer>,<your answer>])
    
    end_time = datetime.datetime.now() - start_time
    print("[Epoch %d/%d] [D loss: %f, acc: %3d%%] [G loss1: %f,loss2: %f] [time:%s]" % (now_iter,all_epoch,D_loss[0],D_loss[1]*100,G_loss[0],G_loss[1],end_time))




plt.gray()
n = 2
r,c=(3,n)
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


