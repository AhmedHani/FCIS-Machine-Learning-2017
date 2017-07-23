from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("./data/", one_hot=True)

NUM_OF_EPOCHS = 15
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# 1- Create the network weights and initialize them by random values
# 2- Create the network biases and initialize them by random values
# 3- Apply feedforward process between the layers and use sigmoid activation in both layers
# 4- Name the output layer values by hidden_to_output
# 4- Uncomment the cost line

#define the cost function
#cost = tf.nn.softmax_cross_entropy_with_logits(logits=hidden_to_output, labels=y)
#cost = tf.reduce_mean(cost)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

errors = []
epochs = []

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(NUM_OF_EPOCHS):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, error = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
            # Compute average loss
            avg_cost += error / total_batch

        errors.append(avg_cost)
        epochs.append(epoch)

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    plt.title("Learning Curve using mean squared error cost function")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Cost")
    plt.plot(epochs, errors)
    plt.show()

    # Test model
    correct_prediction = tf.equal(tf.argmax(hidden_to_output, 1), tf.argmax(y, 1))

    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Test Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
