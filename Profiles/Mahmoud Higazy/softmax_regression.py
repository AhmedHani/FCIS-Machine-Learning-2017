from data_reader.reader import CsvReader
from util import *
import numpy as np
import matplotlib.pyplot as plt
import copy as cp


class SoftmaxRegression(object):
    def __init__(self, learning_rate=0.01, epochs=50):
        self.__epochs= epochs
        self.__learning_rate = learning_rate

    def fit(self, X, y):
        self.w_ = np.zeros((X.shape[1], X.shape[1]))
        self.b = np.ones((1, X.shape[1]))
        self.cost_ = []

        for i in range(self.__epochs):
            # 1- Calculate the net input W^T * x
            z = self.__net_input(X, self.w_, self.b)
            # 2- Get the activation using Softmax function
            h = self.__activation(z)
            # 3- Calculate the gradient
            temp = X.T.dot(y - h)
            # 4- Update the weights and bias using the gradient and learning rate
            self.w_ += self.__learning_rate * temp
            self.b += self.__learning_rate * sum(temp)
            # 5- Uncomment the cost collecting line
            self.cost_.append(self.__cost(self._cross_entropy(output=h, y_target=y)))

    def _cross_entropy(self, output, y_target):
        return -np.sum(np.log(output) * (y_target), axis=1)

    def __cost(self, cross_entropy):
        return 0.5 * np.mean(cross_entropy)

    def __softmax(self, z):
        return (np.exp(z.T) / np.sum(np.exp(z), axis=1)).T

    def __net_input(self, X, W, b):
        return (X.dot(W) + b)

    def __activation(self, X):
        return self.__softmax(X)

    def predict(self, X):
        # 1- Calculate the net input W^T * x
        z = self.__net_input(X, self.w_, self.b)
        # 2- Return the corresponding one-hot vector for each testing item
        h = self.__activation(z)
        ret = []
        for v in h:
            max_index = 0
            for i in range(len(v)):
                if v[i] > v[max_index]:
                    max_index = i
            temp = []
            for i in range(len(v)):
                temp.append(0)
                if i == max_index:
                    temp[i] = 1
            ret.append(temp)
        return ret

reader = CsvReader("./data/Iris.csv")

iris_features, iris_labels = reader.get_iris_data()

print(len(iris_features))
print(len(iris_labels))

iris_features, iris_labels = shuffle(iris_features, iris_labels)
iris_labels = to_onehot(iris_labels)

train_x, train_y, test_x, test_y = iris_features[0:139], iris_labels[0:139], iris_features[139:], iris_labels[139:]
train_x, train_y, test_x, test_y = np.asarray(train_x), np.asarray(train_y), np.asarray(test_x), np.asarray(test_y)

train_x, means, stds = standardize(train_x)
test_x = standardize(test_x, means, stds)

lr = SoftmaxRegression(learning_rate=0.001, epochs=500)
lr.fit(train_x, train_y)

plt.plot(range(1, len(lr.cost_) + 1), np.log10(lr.cost_))
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Softmax Regression - Learning rate 0.02')

plt.tight_layout()
plt.show()

predicted_test = np.asarray(lr.predict(test_x))

print("Test Accuracy: " + str(((sum([np.array_equal(predicted_test[i], test_y[i]) for i in range(0, len(predicted_test))]) / len(predicted_test)) * 100.0)) + "%")
