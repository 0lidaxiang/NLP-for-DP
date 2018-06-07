
# coding: utf-8


from __future__ import print_function, division

import os
import sys
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from scipy.misc import *
from glob import glob

from keras.datasets import mnist
from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam



class Pix2Pix():
    def __init__(self):
        # Input shape
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Configure data loader
        self.dataset_name = 'train'
        
        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()
        
        # Input images and their conditioning images
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # By conditioning on B generate a fake version of A
        fake_A = self.generator(img_B)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([fake_A, img_B])

        self.combined = Model(inputs=[img_A, img_B], outputs=[valid, fake_A])
        self.combined.compile(loss=['mse', 'mae'], loss_weights=[1, 100], optimizer=optimizer)

    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4, bn=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.img_shape)

        # Downsampling
        d1 = conv2d(d0, self.gf, bn=False)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)
        d5 = conv2d(d4, self.gf*8)
        d6 = conv2d(d5, self.gf*8)
        d7 = conv2d(d6, self.gf*8)

        # Upsampling
        u1 = deconv2d(d7, d6, self.gf*8)
        u2 = deconv2d(u1, d5, self.gf*8)
        u3 = deconv2d(u2, d4, self.gf*8)
        u4 = deconv2d(u3, d3, self.gf*4)
        u5 = deconv2d(u4, d2, self.gf*2)
        u6 = deconv2d(u5, d1, self.gf)

        u7 = UpSampling2D(size=2)(u6)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='sigmoid')(u7)

        return Model(d0, output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_A, img_B])

        d1 = d_layer(combined_imgs, self.df, bn=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model([img_A, img_B], validity)

if __name__ == '__main__':
    gan = Pix2Pix()
    print("generator: ")
#     gan.generator.summary()
    print("discriminator: ")
#     gan.discriminator.summary()
#     gan.combined.summary()


def generator_training_Img(real_list_dir,white_list_dir,resize=None,batch_size=32):
    batch_real_img=[]
    batch_white_img=[]
    for _ in range(batch_size):
        random_img_index = np.random.randint(0, 254, size=1)[0]
        real_img =  imread(real_list_dir[random_img_index] , mode='L')
        white_img =  imread(white_list_dir[random_img_index] , mode='L')

        if resize:
            real_img = imresize(real_img,resize)
            white_img = imresize(white_img,resize)
        batch_real_img.append(real_img)
        batch_white_img.append(white_img)
    batch_real_img = np.array(batch_real_img)/127.5-1
    batch_real_img = np.expand_dims(batch_real_img,axis=1)
    batch_white_img = np.array(batch_white_img)/127.5-1
    batch_white_img = np.expand_dims(batch_white_img,axis=3)
    return batch_real_img,batch_white_img

def generator_test_Img(white_list_dir,resize=None ):
    batch_real_img=[]
    batch_white_img=[]
    for i in range(10):
        white_img =  imread(white_list_dir[i] , mode='L')

        if resize:
            white_img = imresize(white_img,resize)
        batch_white_img.append(white_img)
    batch_white_img = np.array(batch_white_img)/127.5-1
    batch_white_img = np.expand_dims(batch_white_img,axis=3)
    return batch_white_img


train_real_data_dir = r'./datasets/train/Real/*'
train_white_data_dir = r'./datasets/train/White/*'

real_list = glob(train_real_data_dir)
train_real_data_list = []
train_real_data_list.extend(real_list)

white_list = glob(train_white_data_dir)
train_white_data_list = []
train_white_data_list.extend(white_list)


epochs = 7000
batch_size_val = 32
all_d_loss = np.zeros(epochs)
all_g_loss = np.zeros(epochs)
    
# Adversarial loss ground truths
valid = np.ones((batch_size_val, 8, 8,1))
fake  = np.zeros((batch_size_val, 8, 8, 1))

for epoch in range(0, epochs):
        start_time = datetime.datetime.now()
        
        ori_img,white_img = generator_training_Img(real_list_dir=train_real_data_list,
                                           white_list_dir=train_white_data_list,
                                           resize=(128,128),
                                           batch_size= batch_size_val)
        imgs_A = ori_img 
        imgs_B = white_img 
        imgs_B = imgs_B.reshape((32,128,128,1))
        imgs_A = imgs_A.reshape((32,128,128,1))
        

        fake_A = gan.generator.predict(imgs_B)
        d_loss_real = gan.discriminator.train_on_batch([imgs_A, imgs_B], valid)
        d_loss_fake = gan.discriminator.train_on_batch([fake_A, imgs_B], fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                
        for i in range(4):
            g_loss = gan.combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])
        
        all_d_loss[epoch] = d_loss[0]
        all_g_loss[epoch] = g_loss[0]
        
        elapsed_time = str(datetime.datetime.now() - start_time)
        print_out = (epoch, epochs, d_loss[0], 100*d_loss[1], g_loss[0],elapsed_time.split(".")[0])
        print ("[Epoch %d/%d]  [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % print_out)
        np.savetxt("all_d_loss.txt", all_d_loss, delimiter=",")
        np.savetxt("all_g_loss.txt", all_g_loss, delimiter=",")


test_white_data_dir = r'./datasets/test/*'
test_white_list = glob(test_white_data_dir)
test_white_data_list = []
test_white_data_list.extend(test_white_list)
test_white_data_list = sorted(test_white_data_list)

print(len(test_white_data_list), test_white_data_list)
test_white_data_list = generator_test_Img( white_list_dir=test_white_data_list, resize=(128,128))

fake_A = gan.generator.predict(test_white_data_list)
gen_imgs = np.concatenate([fake_A])
gen_imgs = 0.5 * gen_imgs
print(gen_imgs.shape)


ids = 0
for img in gen_imgs:
    img = img.reshape((128, 128))
    plt.imsave("res_images/main_test_res_" + str(ids) + ".jpg", img, cmap="gray")
    ids += 1                  
plt.close()   
print("test_data generator predict over.")


def numpy_to_csv(input_image,image_number=10,save_csv_name='predict.csv'):
    save_image=np.zeros([int(input_image.size/image_number),image_number],dtype=np.float32)

    for image_index in range(image_number):
        save_image[:,image_index]=input_image[image_index,:,:].flatten()

    base_word='id'
    df = pd.DataFrame(save_image)
    index_col=[]
    for i in range(n):
        col_word=base_word+str(i)
        index_col.append(col_word)
    df.index.name='index'
    df.columns=index_col
    df.to_csv(save_csv_name)
    print("Okay! numpy_to_csv")

n=10
numpy_to_csv(input_image= gen_imgs,image_number=n,save_csv_name='Predict.csv')


# draw loss 
all_d_loss_txt = np.loadtxt("all_d_loss.txt")
all_g_loss_txt = np.loadtxt("all_g_loss.txt")

# print( all_d_loss_txt.shape, all_d_loss_txt.shape[0])
# print(all_g_loss_txt, all_g_loss_txt.shape, all_g_loss_txt.shape[0])

fig = plt.figure()
ax = plt.axes()
all_d_loss_x = np.linspace(0, 1, all_d_loss_txt.shape[0])
all_g_loss_x = np.linspace(0, 1, all_g_loss_txt.shape[0])

plt.plot(all_g_loss_x, all_g_loss_txt, '-r');  # dotted red, g_loss
plt.plot(all_d_loss_x , all_d_loss_txt , '-g');  # dotted green, d_loss

plt.show()

