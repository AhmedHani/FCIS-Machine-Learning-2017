from session3.data_reader.reader import CsvReader
from session3.util import *
import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression(object):
    def __init__(self, learning_rate=0.01, epochs=50):
        self.__epochs= epochs
        self.__learning_rate = learning_rate

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.__epochs):
            y_ = self.__net_input(X)
            activated_y = self.__activation(y_)
            errors = (y - activated_y)
            neg_grad = X.T.dot(errors)

            self.w_[1:] += self.__learning_rate * neg_grad
            self.w_[0] += self.__learning_rate * errors.sum()

            self.cost_.append(self.__logit_cost(y, self.__activation(y_)))

    def __logit_cost(self, y, y_val):
        logit = -y.dot(np.log(y_val)) - ((1 - y).dot(np.log(1 - y_val)))

        return logit

    def __sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def __net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def __activation(self, X):
        return self.__sigmoid(X)

    def predict(self, X):
        z = self.__net_input(X)

        return np.where(self.__activation(z) >= 0.5, 1, 0)

reader = CsvReader("./data/Iris.csv")

iris_features, iris_labels = reader.get_iris_data()

ignore_verginica = [i for i, v in enumerate(iris_labels) if v == 'Iris-virginica']
iris_features = [v for i, v in enumerate(iris_features) if i not in ignore_verginica]
iris_labels = [v for i, v in enumerate(iris_labels) if i not in ignore_verginica]

print(len(iris_features))
print(len(iris_labels))

iris_features, iris_labels = shuffle(iris_features, iris_labels)
iris_labels = to_onehot(iris_labels)
iris_labels = list(map(lambda v: v.index(max(v)), iris_labels))

train_x, train_y, test_x, test_y = iris_features[0:89], iris_labels[0:89], iris_features[89:], iris_labels[89:]
train_x, train_y, test_x, test_y = np.asarray(train_x), np.asarray(train_y), np.asarray(test_x), np.asarray(test_y)

train_x, means, stds = standardize(train_x)
test_x = standardize(test_x, means, stds)

lr = LogisticRegression(learning_rate=0.1, epochs=1)
lr.fit(train_x, train_y)

plt.plot(range(1, len(lr.cost_) + 1), np.log10(lr.cost_))
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Logistic Regression - Learning rate 0.01')

plt.tight_layout()
plt.show()

predicted_test = lr.predict(test_x)

print("Test Accuracy: " + str(((sum([predicted_test[i] == test_y[i] for i in range(0, len(predicted_test))]) / len(predicted_test)) * 100.0)) + "%")
