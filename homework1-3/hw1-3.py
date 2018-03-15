#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
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



train_1 = np.loadtxt('./data/data.txt')
answerTemp = np.loadtxt('./data/answer.txt')
train_2 = answerTemp.reshape((answerTemp.shape[0],1))

# train_1 = np.array ([[1. , 2. , 3.],
# [3.,4.,5.],
# [8.,5.,7.],
# [7.,1.,8.]])
#
# train_2 = np.array ([[1.],
# [0.],
# [0.],
# [1.]])

input_1 = tf.placeholder(tf.float32, shape = [None, 32])
input_2 = tf.placeholder(tf.float32, shape = [None, 1])

weight_1 = tf.get_variable(name = "weight_1", shape = [32,16], dtype = tf.float32, initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
bias_1 = tf.get_variable(name='bias_1', shape= [16], dtype= tf.float32, initializer= tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
layer_1_output = tf.add(tf.matmul( input_1, weight_1 ), bias_1)

weight_2 = tf.get_variable(name = "weight_2", shape = [16,8], dtype = tf.float32, initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
bias_2 = tf.get_variable(name='bias_2', shape= [8], dtype= tf.float32, initializer= tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
layer_2_output =  tf.add( tf.matmul( layer_1_output, weight_2 ), bias_2)

weight_3 = tf.get_variable(name = "weight_3", shape = [8,4], dtype = tf.float32, initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.05))
bias_3 = tf.get_variable(name='bias_3', shape= [4], dtype= tf.float32, initializer= tf.truncated_normal_initializer(mean=0.0, stddev=0.05))
layer_3_output =  tf.add( tf.matmul( layer_2_output, weight_3 ), bias_3)

weight_last = tf.get_variable(name = "weight_last", shape = [4,1], dtype = tf.float32, initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.05))
bias_last = tf.get_variable(name='bias_last', shape= [1], dtype= tf.float32, initializer= tf.truncated_normal_initializer(mean=0.0, stddev=0.05))
layer_last_output = tf.sigmoid( tf.add( tf.matmul( layer_3_output, weight_last ), bias_last) )

# weight_4 = tf.get_variable(name = "weight_4", shape = [4,2], dtype = tf.float32, initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
# bias_4 = tf.get_variable(name='bias_4', shape= [2], dtype= tf.float32, initializer= tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
# layer_4_output =  tf.add( tf.matmul( layer_3_output, weight_4 ), bias_4)

# weight_last = tf.get_variable(name = "weight_last", shape = [2,1], dtype = tf.float32, initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
# bias_last = tf.get_variable(name='bias_last', shape= [1], dtype= tf.float32, initializer= tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
# layer_last_output = tf.sigmoid( tf.add( tf.matmul( layer_4_output, weight_last ), bias_last) )

loss = tf.losses.mean_squared_error(train_2, layer_last_output)
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(loss)

with tf.Session() as sess:
    #initial
    init = tf.global_variables_initializer()
    sess.run( init )

    loss_arr = []
    accuracy_arr = []
    # train
    for step in range(500):
        loss_value = sess.run(loss, feed_dict = {input_1: train_1, input_2: train_2})
        accuracy_value = sess.run(layer_last_output, feed_dict = {input_1: train_1})

        loss_arr.append(loss_value)
        accuracy_arr.append(accuracy_value)

        if step % 20 == 0:
            print('loss: ', loss_value )
            print('predict: ',  accuracy_value)
        sess.run(train, feed_dict = {input_1: train_1, input_2: train_2})

    fig1 = plt.figure(1)
    fig1.canvas.set_window_title('NLP Homework 1-3')
    plt.figure(1)

    loss_arr_len = len(loss_arr)
    xx = np.linspace(0, loss_arr_len, loss_arr_len)
    plot1 = plt.plot(xx, loss_arr, "b-", label="loss function image")

    plt.title('loss function' ,fontsize=18)
    plt.grid(True)
    plt.legend()
    plt.show()
