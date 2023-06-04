import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(23)


def data_gen(n_samples):
    x = np.sort(np.random.uniform(0, 1, n_samples))
    N = np.random.normal(0, 1, n_samples)
    y = np.sin(2 * np.pi * x) + 0.1 * N
    data = pd.DataFrame()  # DataFrame is a structure
    # that contains two-dimensional data and its corresponding labels.
    data['X'] = x
    data['Y'] = y
    return data


data = data_gen(n_samples=20)


def train_test_split(data):
    train_data = data.sample(frac=0.5, random_state=23)
    # Return a random sample of items from an axis of object
    test_data = data.drop(train_data.index).sample(frac=1.0)
    # Drop specified labels from rows or columns.
    x_train = np.array(train_data['X']).reshape(-1, 1)
    y_train = np.array(train_data['Y']).reshape(-1, 1)
    x_test = np.array(test_data['X']).reshape(-1, 1)
    y_test = np.array(test_data['Y']).reshape(-1, 1)
    return x_train, y_train, x_test, y_test


x_train, y_train, x_test, y_test = train_test_split(data)


def forward(x, w):
    y_pred = w[0]
    for i in range(1, len(w)):
        y_pred += w[i] * x ** i  # yp = w0 + w1*x + w2*x^2 + w3*x^3 .. polynomial
    return y_pred


def loss(y, y_pred):
    loss = np.sqrt(np.mean((y_pred - y) ** 2))
    return loss


def gradient(x, y, y_pred):
    m = x.shape[0]  # nb ligne
    dw = [(2 / m) * np.sum((y_pred - y) * x ** i) for i in range(len(w))]  # os1 os2 ... hta gae w
    return dw


# -------------------------------------------------------------------

b = 1
a = 0.01  # taux d'apprentissage
nw = x_train.shape[0]  # nombre de poids (ligne de x train)
w = [1] + [0] * (nw - 1)  # liste de poids, w[0] = 1, lba9i gae 0
# Training loop
losses = []
for epoch in range(100):
    y_pred = forward(x_train, w)
    dw = gradient(x_train, y_train, y_pred)
    for i in range(len(w)):
        w[i] -= a * dw[i]  # mettre a jour tout les poids

    l = loss(y_train, y_pred)
    losses.append(l)

y_pred = forward(data['X'].array.reshape(-1, 1), w)
fig = plt.figure(figsize=(8, 6))
plt.plot(data['X'], data['Y'], 'yo')
plt.plot(data['X'], y_pred, 'r')
plt.legend(["Data", f"Degree={nw}"])
plt.xlabel('X - Input')
plt.ylabel('y - target / true')
plt.title('Polynomial Regression')
plt.show()
