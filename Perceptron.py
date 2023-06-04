import numpy as np


class Perceptron:

    def __init__(self, activation='step', learning_rate=0.01, n_iterations=1000):
        self.weights = None
        self.activation = activation
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features + 1)

        X = np.hstack((np.ones((n_samples, 1)), X))

        if self.activation == 'step':
            activation_func = lambda x: np.where(x >= 0, 1, 0)
        elif self.activation == 'sigmoid':
            activation_func = lambda x: 1 / (1 + np.exp(-x))
        elif self.activation == 'relu':
            activation_func = lambda x: np.maximum(0, x)
        else:
            raise ValueError('Invalid activation function.')

        for _ in range(self.n_iterations):
            for i in range(n_samples):
                y_pred = activation_func(np.dot(self.weights, X[i]))
                error = y[i] - y_pred
                self.weights += self.learning_rate * error * X[i]

    def predict(self, X):
        n_samples = X.shape[0]
        X = np.hstack((np.ones((n_samples, 1)), X))
        y_pred = np.dot(X, self.weights)

        if self.activation == 'step':
            y_pred = np.where(y_pred >= 0, 1, 0)
        elif self.activation == 'sigmoid':
            y_pred = 1 / (1 + np.exp(-y_pred))
            y_pred = np.where(y_pred >= 0.5, 1, 0)
        elif self.activation == 'relu':
            y_pred = np.maximum(0, y_pred)
            y_pred = np.where(y_pred >= 0.5, 1, 0)
        else:
            raise ValueError('Invalid activation function.')

        return y_pred


X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([0, 0, 0, 1])

model = Perceptron(activation='sigmoid')
model.fit(X_train, y_train)

X_test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_pred = model.predict(X_test)

print(y_pred)  # prints [0, 0, 0, 1]
