#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import tensorflow as tf
import numpy as np

# # 创建一个常量 op, 产生一个 1x2 矩阵. 这个 op 被作为一个节点
# # 加到默认图中.
# #
# # 构造器的返回值代表该常量 op 的返回值.
# matrix1 = tf.constant([[3., 3.]])
#
# # 创建另外一个常量 op, 产生一个 2x1 矩阵.
# matrix2 = tf.constant([[2.],[2.]])
#
# # 创建一个矩阵乘法 matmul op , 把 'matrix1' 和 'matrix2' 作为输入.
# # 返回值 'product' 代表矩阵乘法的结果.
# product = tf.matmul(matrix1, matrix2)

input_1 = tf.placeholder(tf.float32, shape = [None, 3])
input_2 = tf.placeholder(tf.float32, shape = [None, 1])

train_1 = np.array ([[1. , 2. , 3.],
[3.,4.,5.],
[8.,5.,7.],
[7.,1.,8.]])

train_2 = np.array ([[1.],
[0.],
[0.],
[1.]])

weight_1 = tf.get_variable(name = "weight_1", shape = [3,2], dtype = tf.float32, initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
bias_1 = tf.get_variable(name='bias_1', shape= [2], dtype= tf.float32, initializer= tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
layer_1_output = tf.add(tf.matmul( input_1, weight_1 ), bias_1)

weight_2 = tf.get_variable(name = "weight_2", shape = [2,1], dtype = tf.float32, initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
bias_2 = tf.get_variable(name='bias_2', shape= [1], dtype= tf.float32, initializer= tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
layer_2_output = tf.sigmoid( tf.add( tf.matmul( layer_1_output, weight_2 ), bias_2) )

loss = tf.losses.mean_squared_error(train_2, layer_2_output)
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(loss)

with tf.Session() as sess:
    #initial
    init = tf.global_variables_initializer()
    sess.run( init )
    # train
    for step in range(201):
        if step % 20 == 0:
            print('loss: ', sess.run(loss, feed_dict = {input_1: train_1, input_2: train_2}) )
            print('predict: ', sess.run(layer_2_output, feed_dict = {input_1: train_1}) )
        sess.run(train, feed_dict = {input_1: train_1, input_2: train_2})
