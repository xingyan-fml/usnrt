# -*- coding: utf-8 -*-

# Amini, A., Schwarting, W., Soleimany, A., & Rus, D. (2020). Deep evidential regression. 
# Advances in Neural Information Processing Systems, 33, 14927-14937.
# https://proceedings.neurips.cc/paper/2020/hash/aab085461de182608ee9f607f3f7d18f-Abstract.html

import copy
import math
import random
import numpy as np
import pandas as pd
from scipy.stats import norm

import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

#==============================================================================

class model_evidential(nn.Module):
    def __init__(self, in_dim, hiddens):
        super(model_evidential, self).__init__()
        
        self.dims = [in_dim] + list(hiddens) + [4]
        
        self.linears = nn.ModuleList()
        for i in range(1, len(self.dims)):
            self.linears.append(nn.Linear(self.dims[i-1], self.dims[i]))
        
        self.activation = nn.ReLU()
        self.positive = nn.Softplus()
    
    def forward(self, X):
        for i in range(len(self.linears) - 1):
            X = self.activation(self.linears[i](X))
        X = self.linears[-1](X)
        mu, logv, logalpha, logbeta = torch.split(X, 1, dim = 1)
        
        v = self.positive(logv) + 1e-8
        alpha = self.positive(logalpha) + 1
        beta = self.positive(logbeta) + 1e-8
        return mu, v, alpha, beta

#==============================================================================

def NIG_NLL(y, gamma, v, alpha, beta, pi = torch.tensor(math.pi)):
    twoBlambda = 2 * beta * (1 + v)
    nll = 0.5 * torch.log(pi / v) - alpha * torch.log(twoBlambda) \
        + (alpha + 0.5) * torch.log(v * (y - gamma) ** 2 + twoBlambda) \
        + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
    return torch.mean(nll)

def NIG_Reg(y, gamma, v, alpha, beta):
    error = torch.abs(y - gamma)
    evi = 2 * v + alpha
    reg = error * evi
    return torch.mean(reg)

def load_array(data_arrays, batch_size, is_train = True):
    dataset = TensorDataset(*data_arrays)
    return DataLoader(dataset, batch_size, shuffle = is_train)

def initialize(model, weight_seed):
    torch.manual_seed(weight_seed)
    for linear in model.linears:
        nn.init.xavier_normal_(linear.weight)
        nn.init.constant_(linear.bias, 0.0)

#==============================================================================

def fit_evidential(data, y, Lambda, hidden_dims, num_epochs, batch_size, lr, train_prop, set_seed):
    model = model_evidential(data.shape[1] - 1, hidden_dims)
    
    random.seed(set_seed)
    train_index = random.sample(range(len(data)), math.floor(len(data) * train_prop))
    valid_index = list(set(range(len(data))) - set(train_index))
    
    X, Y = data.drop(y, axis = 1), data[y]
    X = torch.tensor(X.values, dtype = torch.float32)
    Y = torch.tensor(Y.values, dtype = torch.float32).reshape(-1, 1)
    x_train, y_train = X[train_index], Y[train_index]
    x_valid, y_valid = X[valid_index], Y[valid_index]
    
    initialize(model, set_seed)
    optimizer = optim.Adam(model.parameters(), lr = lr)
    if batch_size < len(x_train):
        train_iter = load_array((x_train, y_train), batch_size = batch_size)
    else:
        train_iter = [(x_train, y_train)]
    
    best_valid = 1e38
    epoch_valids = []
    
    for epoch in range(num_epochs):
        batch_valids = []
        for x_batch, y_batch in train_iter:
            model.train()
            mu, v, alpha, beta = model(x_batch)
            nll_loss = NIG_NLL(y_batch, mu, v, alpha, beta)
            reg_loss = NIG_Reg(y_batch, mu, v, alpha, beta)
            loss = nll_loss + Lambda * reg_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            model.eval()
            with torch.no_grad():
                muV, vV, alphaV, betaV = model(x_valid)
                valid_nll_loss = NIG_NLL(y_valid, muV, vV, alphaV, betaV)
                valid_reg_loss = NIG_Reg(y_valid, muV, vV, alphaV, betaV)
                valid_loss = (valid_nll_loss + Lambda * valid_reg_loss).detach().numpy()
                batch_valids.append(valid_loss)
                if valid_loss < best_valid:
                    best_valid = valid_loss
                    best_dict = copy.deepcopy(model.state_dict())
        
        print("epoch {}:".format(epoch), np.mean(batch_valids))
        epoch_valids.append(np.mean(batch_valids))
        if len(epoch_valids) >= 25 and np.mean(epoch_valids[-5:]) >= np.mean(epoch_valids[-25:-5]):
            break
    
    model.load_state_dict(best_dict)
    model.eval()
    
    return model

def predict_evidential(model, x_test, y_test):
    if not torch.is_tensor(x_test):
        x_test = torch.tensor(x_test.values, dtype = torch.float32)
    if not torch.is_tensor(y_test):
        y_test = torch.tensor(y_test.values, dtype = torch.float32).reshape(-1, 1)
    
    model.eval()
    with torch.no_grad():
        mu, v, alpha, beta = model(x_test)
        nll_loss = NIG_NLL(y_test, mu, v, alpha, beta)
        mu, v = mu.detach().numpy().reshape(-1), v.detach().numpy().reshape(-1)
        alpha = alpha.detach().numpy().reshape(-1)
        beta = beta.detach().numpy().reshape(-1)
        nll_loss = nll_loss.detach().numpy().item()
    
    return mu, v, alpha, beta, nll_loss
