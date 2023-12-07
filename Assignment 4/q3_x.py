"""
Solution to MME 9656 Assigment 4 Question 3

Author: Vlad Pac
Due Date: December 6, 2023
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from sklearn.model_selection import train_test_split


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

sim_time = 100
dt = 0.1

df = pd.read_csv('Assignment 4\duffing_100s.csv', header=None, usecols=[0])
x_values = df.values.astype('float32')
# print(df.head().to_string())

test_size = 0.2
train, test = train_test_split(x_values, test_size=test_size, shuffle=False)
# train, test = df[:8000], df[8000:]
# print(type(train))

# t = np.arange(0, sim_time, dt)
# t_train = int((1-test_size)*len(t))
# plt.plot(t, df)
# plt.plot(t[:t_train], train)
# plt.plot(t[t_train:], test)
# plt.show()

def create_dataset(dataset, lookback):
    tensor_size = len(dataset)-lookback
    # X, y = np.zeros(tensor_size), np.zeros(tensor_size)

    X, y = [], []
    for i in range(tensor_size):
        feature = dataset[i:i+lookback]
        target = dataset[i+1:i+lookback+1]
        X.append(feature)
        y.append(target)
    X = np.array(X)
    y = np.array(y)
    return torch.FloatTensor(X).to(device), torch.FloatTensor(y).to(device)

lookback = 1
X_train, y_train = create_dataset(train, lookback)
X_test, y_test = create_dataset(test, lookback)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

input_size = 1
hidden_size = 50
num_layers = 1
output_size = 1

class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.linear = nn.Linear(in_features=hidden_size,
                                out_features=output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x
    
model = Model()
model.to(device)
optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()
loader = data.DataLoader(data.TensorDataset(X_train, y_train),
                         shuffle=True,
                         batch_size=8)

n_epochs = 50
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if epoch % 100 == 0:
        model.eval()
        with torch.no_grad():
            y_pred = model(X_train)
            train_rmse = np.sqrt(loss_fn(y_pred, y_train).detach().cpu().numpy())
            y_pred = model(X_test)
            test_rmse = np.sqrt(loss_fn(y_pred, y_test).detach().cpu().numpy())
        print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % \
                (epoch, train_rmse, test_rmse))

with torch.no_grad():
    train_plot = np.ones_like(x_values) * np.nan
    y_pred = model(X_train)
    y_pred = y_pred[:, -1, :]
    train_plot[lookback:len(train)] = model(X_train).detach().cpu().numpy()[:, -1, :]

    test_plot = np.ones_like(x_values) * np.nan
    test_plot[len(train)+lookback:len(x_values)] = model(X_test).detach().cpu().numpy()[:, -1, :]

plt.plot(x_values, c='b')
plt.plot(train_plot, c='r')
plt.plot(test_plot, c='g')
plt.show()