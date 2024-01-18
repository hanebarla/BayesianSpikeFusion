#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

# LSMで作った式のy座標を求める関数
def quation_LSM(W, X) :
    len_X = len(X)
    len_W = len(W)
    Y = []
    for i in range(len_X) :
        y = 0
        for j in range(len_W) :
            y += W[len_W-1-j] * pow(X[i], j)
        Y.append(y)
    return Y

# 正規化LSM実行関数(戻り値:重みベクトルW)
def regularization_LSM(data, N, LAMBDA) :
    # X・coe = Y
    X = np.zeros([N,N])
    Y = np.zeros([N,1])

    temp = 0
    for i in range(N):
        for j in range(N):
            temp = 0
            for d in data :
                temp += pow(d[0],2*(N-1)-i-j)
            if 2*(N-1)-i-j == 0 :
                temp = len(data)
            X[i][j] = temp

    for i in range(N):
        temp = 0
        for d in data :
            temp += pow(d[0],N-1-i) * d[1]
            Y[i] = temp

    for i in range(N):
        X[i][i] += LAMBDA

    # coe = inverseX・Y
    # coe:重みベクトル
    coe = np.dot(np.linalg.inv(X),Y)
    coe = coe[:,0]

    # error:二乗和誤差
    error = 0
    maked_data = data
    for i in data :
        temp = 0
        for j in range(N) :
            temp += coe[N-1-j] * pow(i[0], j)

        error += pow(i[1] - temp, 2.0)

    #error = error / len(data)
    error /= 2
    error = np.sqrt(error)

    return error, coe

# LSM実行関数(戻り値:重みベクトルW)
def LSM(data, N) :

    # X・coe = Y
    X = np.zeros([N,N])
    Y = np.zeros([N,1])

    temp = 0
    for i in range(N):
        for j in range(N):
            temp = 0
            for d in data :
                temp += pow(d[0],2*(N-1)-i-j)
            if 2*(N-1)-i-j == 0 :
                temp = len(data)
            X[i][j] = temp

    for i in range(N):
        temp = 0
        for d in data :
            temp += pow(d[0],N-1-i) * d[1]
            Y[i] = temp

    # coe = inverseX・Y
    # coe:重みベクトル
    coe = np.dot(np.linalg.inv(X),Y)
    coe = coe[:,0]

    # error:二乗和誤差
    error = 0
    maked_data = data
    for i in data :
        temp = 0
        for j in range(N) :
            temp += coe[N-1-j] * pow(i[0], j)

        error += pow(i[1] - temp, 2.0)

    #error = error / len(data)
    error /= 2
    error = np.sqrt(error)

    return error, coe

# class gauss_func(torch.nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.amp = torch.nn.Parameter(torch.ones(1,))
#         self.mu = torch.nn.Parameter(torch.ones(1,))
#         self.var = torch.nn.Parameter(torch.ones(1,))
#         self.relu = torch.nn.ReLU()
#
#     def forward(self, x):
#         x = torch.exp(-torch.pow((x-self.relu(self.mu)), 2)/(2*self.relu(self.var)))
#         x = x / torch.sqrt(2*math.pi*self.relu(self.var))
#         y = self.amp * x
#
#         return y

def gauss_func(x, amp, mu, var, offset):
    y = np.exp(-np.power(x-mu, 2)/(2*var))
    y = amp * y / np.sqrt(2*np.pi*var)
    y = y + offset

    return y

def log_gauss_func(x, amp, mu, var, offset):
    y = np.exp(-np.power(np.log(x)-mu, 2)/(2*var))
    y = amp * y / (np.sqrt(2*np.pi*var) * x)
    y = y + offset

    return y

def gauss_LSM(x_unit, y):
    normalize_c = y.max()
    y_unit = y / normalize_c

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # x_torch = torch.from_numpy(x_unit.astype(np.float32)).clone()
    # y_torch = torch.from_numpy(y_unit.astype(np.float32)).clone()

    # model = gauss_func()
    # model.to(device)
    # model.train()
    # optim = torch.optim.SGD(model.parameters(), lr=0.000001, momentum=0.9, nesterov=True)

    # amps = np.array([0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1])
    # mus = np.array([1, 0.1, 0.01, 0.001, 0.0001])
    # vars = np.array([0.001, 0.0001])
    # offsets = np.array([1, 0.1, 0.2, 0.3, 0.4, 0.5])

    amps = np.logspace(-4, 0, 30)
    mus = np.logspace(-4, 0, 30)
    vars = np.logspace(-4, 0, 30)
    offsets = np.linspace(0, 0.5, 20)

    conditions = np.array(np.meshgrid(amps, mus, vars, offsets)).T.reshape(-1, 4)
    print(conditions.shape)  # 0: amps, 1: mus, 2: vars
    search_num = conditions.shape[0]

    best_loss = 10000
    best_amp = -1
    best_mu = -1
    best_var = -1
    best_offset = -1

    for i in trange(search_num):
        amp, mu, var, offset = conditions[i, 0], conditions[i, 1], conditions[i, 2], conditions[i, 3]
        x = gauss_func(x_unit, amp, mu, var, offset)

        loss = np.mean(np.power(y_unit-x, 2))
        # print(loss)
        if loss < best_loss:
            best_loss = loss
            best_amp, best_mu, best_var, best_offset = amp, mu, var, offset

        # input_x = x_torch.detach().clone().to(device)
        # label = y_torch.detach().clone().to(device)
        # y_output = model(input_x)
        # loss = torch.mean((y_output-label)*(y_output-label))

        # loss.backward()
        # optim.step()

    print("best: amp: {}".format(best_amp))
    print("best: mu: {}".format(best_mu))
    print("best: var: {}".format(best_var))
    print("best: offset: {}".format(best_offset))

    best_conditions = {
        "MSE": best_loss,
        "amp": best_amp * normalize_c,
        "mu": best_mu,
        "var": best_var,
        "offset": best_offset
    }
    print(best_conditions)

    return best_conditions