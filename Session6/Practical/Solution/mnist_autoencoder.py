import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./data/", one_hot=True)

# Parameters
learning_rate = 0.1
training_epochs = 100
batch_size = 256
display_step = 1
examples_to_show = 10

# Network Parameters
n_hidden_1 = 256 # 1st layer num features
n_hidden_2 = 128 # 2nd layer num features
n_input = 784 # MNIST data input (img shape: 28*28)

X = tf.placeholder("float", [None, n_input])

input_to_hidden_encoder_weights = tf.Variable(tf.random_normal([n_input, n_hidden_1]))
input_to_hidden_encoder_bias = tf.Variable(tf.random_normal([n_hidden_1]))

hidden_to_hidden_encoder_weights = tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]))
hidden_to_hidden_encoder_bias = tf.Variable(tf.random_normal([n_hidden_2]))

hidden_to_hidden_decoder_weights = tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1]))
hidden_to_hidden_decoder_bias = tf.Variable(tf.random_normal([n_hidden_1]))

hidden_to_output_decoder_weights = tf.Variable(tf.random_normal([n_hidden_1, n_input]))
hidden_to_output_decoder_bias = tf.Variable(tf.random_normal([n_input]))


input_to_hidden_encode = tf.matmul(X, input_to_hidden_encoder_weights) + input_to_hidden_encoder_bias
input_to_hidden_encode = tf.nn.sigmoid(input_to_hidden_encode)

hidden_to_hidden_encode = tf.matmul(input_to_hidden_encode, hidden_to_hidden_encoder_weights) + hidden_to_hidden_encoder_bias
hidden_to_hidden_encode = tf.nn.sigmoid(hidden_to_hidden_encode)

hidden_to_hidden_decode = tf.matmul(hidden_to_hidden_encode, hidden_to_hidden_decoder_weights) + hidden_to_hidden_decoder_bias
hidden_to_hidden_decode = tf.nn.sigmoid(hidden_to_hidden_decode)

hidden_to_output_decode = tf.matmul(hidden_to_hidden_decode, hidden_to_output_decoder_weights) + hidden_to_output_decoder_bias
hidden_to_output_decode = tf.nn.sigmoid(hidden_to_output_decode)

# Prediction
y_pred = hidden_to_output_decode
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    total_batch = int(mnist.train.num_examples/batch_size)
    # Training cycle
    for epoch in range(training_epochs):
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(c))

    print("Optimization Finished!")

    # Applying encode and decode over test set
    encode_decode = sess.run(
        y_pred, feed_dict={X: mnist.test.images[:examples_to_show]})
    # Compare original images with their reconstructions
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
    f.show()
    plt.draw()
    plt.waitforbuttonpress()