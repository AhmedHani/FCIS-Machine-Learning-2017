from session3.data_reader.reader import CsvReader
from session3.util import *
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
            y_ = self.__net_input(X, self.w_, self.b)
            activated_y = self.__activation(y_)
            errors = (y - activated_y)
            neg_grad = X.T.dot(errors)

            self.w_ += self.__learning_rate * neg_grad
            self.b += self.__learning_rate * errors.sum()

            self.cost_.append(self.__cost(self._cross_entropy(output=activated_y, y_target=y)))

    def __logit_cost(self, y, y_val):
        logit = -y.dot(np.log(y_val)) - ((1 - y).dot(np.log(1 - y_val)))

        return logit

    def _cross_entropy(self, output, y_target):
        return -np.sum(np.log(output) * (y_target), axis=1)

    def __cost(self, cross_entropy):
        return 0.5 * np.mean(cross_entropy)

    def __sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def __softmax(self, z):
        return (np.exp(z.T) / np.sum(np.exp(z), axis=1)).T

    def __net_input(self, X, W, b):
        return (X.dot(W) + b)

    def __activation(self, X):
        return self.__softmax(X)

    def predict(self, X):
        z = self.__net_input(X, self.w_, self.b)
        activated_z = self.__softmax(z)
        vec = [0]*len(activated_z[1])
        max_indices = list(map(lambda v: list(v).index(max(v)), activated_z))

        for i in range(0, len(max_indices)):
            copy_vec = cp.copy(vec)
            copy_vec[max_indices[i]] = 1
            max_indices[i] = copy_vec

        return max_indices

reader = CsvReader("./data/Iris.csv")

iris_features, iris_labels = reader.get_iris_data()

print(len(iris_features))
print(len(iris_labels))

iris_features, iris_labels = shuffle(iris_features, iris_labels)
iris_labels = to_onehot(iris_labels)

train_x, train_y, test_x, test_y = iris_features[0:89], iris_labels[0:89], iris_features[89:], iris_labels[89:]
train_x, train_y, test_x, test_y = np.asarray(train_x), np.asarray(train_y), np.asarray(test_x), np.asarray(test_y)

train_x, means, stds = standardize(train_x)
test_x = standardize(test_x, means, stds)

lr = SoftmaxRegression(learning_rate=0.02, epochs=800)
lr.fit(train_x, train_y)

plt.plot(range(1, len(lr.cost_) + 1), np.log10(lr.cost_))
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Softmax Regression - Learning rate 0.02')

plt.tight_layout()
plt.show()

predicted_test = np.asarray(lr.predict(test_x))

print("Test Accuracy: " + str(((sum([np.array_equal(predicted_test[i], test_y[i]) for i in range(0, len(predicted_test))]) / len(predicted_test)) * 100.0)) + "%")
