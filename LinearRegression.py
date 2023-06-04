# matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import math
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


print(data)


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
# print(x_train, y_train, x_test, y_test)

# Degree 0 means y=b i.e there is only bias
b = 1
w = [0, 1]


# -------------------
def forward_0(x):
    return w[0] + w[1] * x


# -----------------------------------------
def loss(y, y_pred):
    loss = np.sqrt(np.mean((y_pred - y) ** 2))
    return loss


# ------------------------------------------
def gradient_0(x, y, y_pred):
    m = x.shape[0]
    dw0 = (2 / m) * np.sum((y_pred - y))
    dw1 = (2 / m) * np.sum((y_pred - y) * x)
    return dw0, dw1


# -------------------------------------------------------------------
a = 0.01  # taux d'apprentissage
# Training loop
losses0 = []
for epoch in range(100):
    losses = []
    # -------------------------
    y_pred = forward_0(x_train)  # Yp = w0 + w1*x
    # -----------------------------------------
    dw0, dw1 = gradient_0(x_train, y_train, y_pred)
    w[0] = w[0] - a * dw0
    w[1] = w[1] - a * dw1
    # ----------------------
    y_pred = forward_0(x_train)  # Yp = w0 + w1*x

    l = loss(y_train, y_pred)
    losses.append(l)
    losses0.append(l)
    if epoch % 10 == 0 or epoch == 99:
        print("progress:", epoch, "w0=", w[0], "w1=", w[1], "loss=", np.mean(losses))

y_pred = forward_0(data['X'].array.reshape(-1, 1))
fig = plt.figure(figsize=(8, 6))
plt.plot(data['X'], data['Y'], 'yo')
plt.plot(data['X'], y_pred, 'r')
plt.legend(["Data", "Degree=1"])
plt.xlabel('X - Input')
plt.ylabel('y - target / true')
plt.title('Linear Regression')
plt.show()
