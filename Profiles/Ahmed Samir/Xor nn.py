import tensorflow as tf
import matplotlib.pyplot as plt #we will use it to draw the learning curve after training the network

train_x = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
train_y = [[0], [1], [1], [0]]

INPUT_NEURONS = 2
HIDDEN_NEURONS = 3
OUTPUT_NEURONS = 1

NUM_OF_EPOCHS = 100000

"""(tf.float32, [None, 2]) specifies the datatype and the dimensions of the data.
#Since we don't know the number of the training data, we make it None which means it accepts any from the user.
#2 specifies that we have 2 input bits
"""
x = tf.placeholder(tf.float32, [None, 2])
y_target = tf.placeholder(tf.float32, [None, 1])
bias=tf.Variable([1.0,1.0,1.0])
"""
1- Create the Input-to-hidden weights and bias matrices from the given figure. 
They should be Variable datatype because they will be changed during the learning process
"""
# Write your code here
weights1=tf.Variable([[-0.99,1.05,.19],[-0.43,-0.44,-0.30]])
weights2=tf.Variable([[0.18],[1.11],[-0.26]])
"""
2- Get the values of the hidden layer by multiplying the features with the weight matrix [Input to Hidden feedforward]
Apply the hidden layer activation to the multiplication result
"""

# Write your code here
res=tf.matmul(x,weights1)+bias
"""
3- Create the hidden-to-output weights and bias matrices from the given figure. 
They should be Variable datatype because they will be changed during the learning process
"""
res=tf.nn.sigmoid(res)
# Write your code here
"""
4- Get the values of the output layer by multiplying the hidden layer with the weight matrix [Hidden to Output feedforward]
Apply the output layer activation to the multiplication result
"""
val=tf.Variable([1.0])
res1=tf.matmul(res,weights2)+val

# Write your code here
res1=tf.nn.sigmoid(res1)

mean_squared_error = 0.5 * tf.reduce_sum((tf.square(res1 - y_target)))
train = tf.train.GradientDescentOptimizer(0.1).minimize(mean_squared_error)


"""
Initiate a Tensorflow graph and session variables
"""
session = tf.Session()
session.run(tf.initialize_all_variables())

errors = []
epochs = []

for i in range(0, NUM_OF_EPOCHS):
    session.run(train, feed_dict={x: train_x, y_target: train_y})

    if i % 10 == 0:
        print("Iteration number: ", i, "\n")
        error = session.run(mean_squared_error, feed_dict={x: train_x, y_target: train_y})
        print("Cost: ", error, "\n")
        errors.append(error)
        epochs.append(i)

        if error < 0.01:
            plt.title("Learning Curve using mean squared error cost function")
            print("Cost: ", error, "\n")
            plt.xlabel("Number of Epochs")
            plt.ylabel("Cost")
            plt.plot(epochs, errors)
            plt.show()

            break
