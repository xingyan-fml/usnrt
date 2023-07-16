# -*- coding: utf-8 -*-
"""
Authors: Wenxuan Ma & Xing Yan @ RUC
mawenxuan@ruc.edu.cn
xingyan@ruc.edu.cn
"""

import copy
import math
import random
import numpy as np
import pandas as pd

import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

#==============================================================================

class model_nn(nn.Module):
    def __init__(self, in_dim, hiddens, activation):
        super(model_nn, self).__init__()
        
        self.dims = [in_dim] + list(hiddens) + [1]
        
        self.linears = nn.ModuleList()
        for i in range(1, len(self.dims)):
            self.linears.append(nn.Linear(self.dims[i-1], self.dims[i]))
        
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
    
    def forward(self, X):
        for i in range(len(self.linears) - 1):
            X = self.activation(self.linears[i](X))
        X = self.linears[-1](X)
        return X

#==============================================================================

criterion = nn.MSELoss()

def load_array(data_arrays, batch_size, is_train = True):
    dataset = TensorDataset(*data_arrays)
    return DataLoader(dataset, batch_size, shuffle = is_train)

def initialize(model, weight_seed):
    torch.manual_seed(weight_seed)
    for linear in model.linears:
        nn.init.xavier_normal_(linear.weight)
        nn.init.constant_(linear.bias, 0.0)

#==============================================================================

def fit_nn(data, var_cont, var_cate, y, hidden_dims, num_epochs, batch_size, lr, train_prop, set_seed):
    model = model_nn(data.shape[1] - 1, hidden_dims, 'relu')
    
    random.seed(set_seed)
    train_index = random.sample(range(len(data)), math.floor(len(data) * train_prop))
    valid_index = list(set(range(len(data))) - set(train_index))
    
    X, Y = data.drop(y, axis = 1), data[y]
    x_train, y_train = X.iloc[train_index], Y.iloc[train_index]
    x_valid, y_valid = X.iloc[valid_index], Y.iloc[valid_index]
    
    X = torch.tensor(X.values, dtype = torch.float32)
    Y = torch.tensor(Y.values, dtype = torch.float32).reshape(-1, 1)
    x_train = torch.tensor(x_train.values, dtype = torch.float32)
    y_train = torch.tensor(y_train.values, dtype = torch.float32).reshape(-1, 1)
    x_valid = torch.tensor(x_valid.values, dtype = torch.float32)
    y_valid = torch.tensor(y_valid.values, dtype = torch.float32).reshape(-1, 1)
    
    initialize(model, set_seed)
    optimizer = optim.Adam(model.parameters(), lr = lr)
    if batch_size < len(x_train):
        train_iter = load_array((x_train, y_train), batch_size = batch_size)
    else:
        train_iter = [(x_train, y_train)]
    
    best_valid = 1e8
    epoch_valids = []
    
    for epoch in range(num_epochs):
        batch_valids = []
        for x_batch, y_batch in train_iter:
            model.train()
            out = model(x_batch)
            loss = criterion(out, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            model.eval()
            with torch.no_grad():
                valid_loss = criterion(model(x_valid), y_valid).detach().numpy()
            batch_valids.append(valid_loss)
            if valid_loss < best_valid:
                best_valid = valid_loss
                best_dict = copy.deepcopy(model.state_dict())
        
        print("epoch {}:".format(epoch), np.mean(batch_valids))
        epoch_valids.append(np.mean(batch_valids))
        if len(epoch_valids) >= 25 and np.mean(epoch_valids[-5:]) > np.mean(epoch_valids[-25:-5]):
            break
    
    model.load_state_dict(best_dict)
    model.eval()
    valid_loss = criterion(model(x_valid), y_valid).detach().numpy()
    print("final:", valid_loss, best_valid)
    Y_hat = model(X).detach().numpy()
    residual = Y.detach().numpy().reshape(-1) - Y_hat.reshape(-1)
    sigma2 = np.mean(residual ** 2)
    
    return model, residual, sigma2

#==============================================================================

if __name__ == "__main__":
    
    N = 10000
    np.random.seed(12)
    data_x = np.random.randn(N, 3)
    data_y = np.dot(data_x, np.array([[0.1], [0.3], [-0.2]])) + 0.5 + 0.2 * np.random.randn(N, 1)
    
    data = pd.DataFrame(data_x, columns = range(data_x.shape[1]))
    data['y'] = data_y
    
    hidden_dims = [16, 4]
    model, residual, sigma2 = fit_nn(data, data.columns[:-1], [], 'y', hidden_dims,
                num_epochs=100, batch_size=64, lr=0.01, train_prop=0.8, set_seed=10)
    
    pred = model(torch.tensor([1, 2, 3], dtype = torch.float32))
    print(pred.detach().numpy().item())



