import tensorflow as tf
import numpy as np

def add_layer(inputs, input_tensor, output_tensor, activation_function=None):
    with tf.name_scope("layer"):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([input_tensor, output_tensor]), name='Weight')
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, output_tensor]), name='Bias')
        with tf.name_scope('neuron'):
            neuron_out = tf.add(tf.matmul(inputs, Weights), biases)
        if activation_function is None:
            outputs = neuron_out
        else:
            outputs = activation_function(neuron_out)
    return outputs


input_data = np.random.rand(10)
input_data = input_data.reshape(len(input_data), 1)
true_label = input_data + 5

input_feed = tf.placeholder(tf.float32, shape = [10, 1])
Label_feed = tf.placeholder(tf.float32, shape = [None, 1])

hidden_layer = add_layer(input_feed, 1, 100, activation_function = None)

output_layer = add_layer(hidden_layer, 100, 1, activation_function = None)

loss = tf.reduce_mean(tf.reduce_sum(Label_feed - output_layer))
train_epoch = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for step in range(100):
    sess.run(train_epoch, feed_dict = {input_feed: input_data, Label_feed: true_label})
    if step % 5 == 0:
        print(sess.run(loss, feed_dict = {input_feed: input_data, Label_feed: true_label}))

sess.close()
