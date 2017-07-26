from data_reader.reader import CsvReader
from util import *
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
            # 1- Calculate the net input W^T * x
            z = self.__net_input(X)
            # 2- Get the activation using Sigmoid function
            h = self.__activation(z)
            # 3- Calculate the gradient
            temp = X.T.dot(y - h)
            # 4- Update the weights and bias using the gradient and learning rate
            self.w_[1:] += self.__learning_rate * temp
            self.w_[0] += self.__learning_rate * sum(temp)
            # 5- Uncomment the cost collecting line
            self.cost_.append(self.__logit_cost(y, self.__activation(z)))

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
        # 1- Calculate the net input W^T * x
        z = self.__net_input(X)
        # 2- Return the activated values (0 or 1 classes)
        h = self.__activation(z)
        return np.where(self.__activation(z) >= 0.5, 1, 0)

reader = CsvReader("./data/titanic/train.csv")

titanic_features, titanic_labels = reader.get_titanic_data()

print(len(titanic_features))
print(len(titanic_labels))

titanic_labels = to_onehot(titanic_labels)
titanic_labels = list(map(lambda v: v.index(max(v)), titanic_labels))

train_x, train_y, test_x, test_y = titanic_features[0:712], titanic_labels[0:712], titanic_features[0:], titanic_labels[0:]
train_x, train_y, test_x, test_y = np.asarray(train_x), np.asarray(train_y), np.asarray(test_x), np.asarray(test_y)

train_x, means, stds = standardize(train_x)
test_x = standardize(test_x, means, stds)

lr = LogisticRegression(learning_rate=0.1, epochs=50)
lr.fit(train_x, train_y)

plt.plot(range(1, len(lr.cost_) + 1), np.log10(lr.cost_))
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Logistic Regression - Learning rate 0.1')

plt.tight_layout()
plt.show()

predicted_test = lr.predict(test_x)

print("Test Accuracy: " + str(((sum([predicted_test[i] == test_y[i] for i in range(0, len(predicted_test))]) / len(predicted_test)) * 100.0)) + "%")
