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
from sklearn.model_selection import train_test_split

sim_time = 100
dt = 0.01

df = pd.read_csv('Assignment 4\duffing_100s.csv', header=None)
# print(df.head().to_string())

test_size = 0.2
train, test = train_test_split(df, test_size=test_size, shuffle=False)
# print(type(train))

t = np.arange(0, sim_time, dt)
t_train = int((1-test_size)*len(t))

# plt.plot(t, df)
# plt.plot(t[:t_train], train)
# plt.plot(t[t_train:], test)
# plt.show()

def create_dataset(dataset, lookback):
    tensor_size = len(dataset)-lookback
    X, y = np.zeros(tensor_size), np.zeros(tensor_size)
    
    for i in range(tensor_size):
        feature = dataset[i:i+lookback]
        target = dataset[i+1:i+lookback+1]
        X[i] = feature



input_size = 3
hidden_size = 20
num_layers = 1
output_size = 3

class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size,
                            num_layers=num_layers)
        self.linear = nn.Linear(in_features=hidden_size,
                                out_features=output_size)

    def forward(self, x):
        x = self.lstm(x)
        x = self.linear(x)
        return x
    
model = Model()
