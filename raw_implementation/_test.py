# -*- coding: utf-8 -*-
"""_test.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1BMUhJQ_E165XvuY0bi2bcE6C3r2Bugh3
"""

from _model import model
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
import numpy as np
from torch.utils.data.dataloader import DataLoader
from _utils import utils
from datetime import datetime
now = datetime.now()
from dataset.electricity_dataset import electricity_pre
import _paper_params

from pandas import read_csv
import pandas
from datetime import datetime
import os
from matplotlib import pyplot

import warnings
warnings.filterwarnings('ignore')

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def date_cv(x):
    return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

dataset = read_csv('dataset/elec.txt', sep=";", decimal=",", header=None, skiprows=[0], parse_dates = [[0]], date_parser=date_cv)
dataset.set_index("0", inplace=True)
weekday = pandas.Series(dataset.index[:].weekday, dtype='int32')
hour = pandas.Series(dataset.index[:].hour, dtype='int32')
minute = pandas.Series(dataset.index[:].minute, dtype='int32')

dataset.insert(len(dataset.columns), 'weekday', weekday.values)
dataset.insert(len(dataset.columns), 'hour', hour.values)
dataset.insert(len(dataset.columns), 'min', minute.values)
dataset.index.name = 'date'
dataset = dataset[35042:]
print(dataset.head(5))

values = dataset.values
groups = [0, 1, 2, 3, 4, 5, 6, 7]
i = 1
fig = pyplot.figure()
for group in groups:
    pyplot.subplot(len(groups), 1, i)
    pyplot.plot(values[20000:24000, group])
    pyplot.title(dataset.columns[group], y=0.5, loc='right')
    i += 1
fig.subplots_adjust(hspace=0.5)
pyplot.show()
dataset.to_csv('dataset/electricity.csv')

params = _paper_params._params_
dataset = electricity_pre(csv_file=params['data_file'], dev=device, ctx_win_len=params['ctx_win_len'], num_time_indx=params['num_time_idx'])

num_covariates = params['num_covariates']
num_time_idx = params['num_time_idx']
num_targets = params['num_targets']
input_dim = num_time_idx + num_targets + num_covariates
'''
Size of the entire context window utilized during training
can be divided into 2 sections - first: conditioning context with input data points and network predictions conditioned on real data; 
and other one is  prediction context, where network predictions are conditioned on network output from the ealrier steps.
'''
ctx_win_len = params['ctx_win_len']
cond_win_len = params['cond_win_len']
pred_win_len = ctx_win_len - cond_win_len - 1
batch_size = params['batch_size']

model = model(num_lstms=params['num_lstms'], input_dim=input_dim, output_dim=params['num_targets'],hidden_dim=params['hidden_dim']).to(device)
optimizer = optim.Adam(model.parameters(), params['lr'])
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=params['lr_step_size'], gamma=params['lr_gamma'])

train_sampler, test_sampler = dataset.get_train_test_samplers(params['train_test_split'])
train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, shuffle=False, num_workers=0)
test_dataloader = DataLoader(dataset, batch_size=1, sampler=test_sampler,shuffle=False, num_workers=0)
max_batches_per_epoch = params['max_batches_per_epoch']
num_epochs = params['num_epochs']

losses = []
batch_num = 1
utils = TSUtils(device, params)
# Trainin loop
def train():
    for epoch in range(0, num_epochs):
        batch_num = 0
        for i, batch in enumerate(train_dataloader):
            if batch_num > max_batches_per_epoch:
                break    
            input, target, covariates = utils.split_batch(batch) #splitting batches : issue with sequence
            #conditioning window input,Prediction window , model op conditioned on previous values
            # model output for the previous step
            input_cond = input[:, 0:cond_win_len, :]
            input_cond, covariates = utils.scale(input_cond, covariates)
            optimizer.zero_grad()
            #forward pass
            out = model(input_cond, covariates, future=pred_win_len)
            out = utils.invert_scale(out, probabalistic=True)
            #rescale
            loss = model.NLL(out, target)
            #loss calc
            loss.backward()
            scheduler.step()
            optimizer.step()
            loss = loss.item()
            losses.append(loss)
            print(f"epoch:{epoch} /{num_epochs}| batch:{batch_num}:loss:{loss}")
            batch_num = batch_num + 1
    return model, losses

def plot_loss(losses):
    plt.figure()
    plt.title('Loss progression')
    plt.xlabel('batch no')
    plt.ylabel('loss')
    plt.plot(np.arange(len(losses)), losses, 'g', linewidth=2.0)
    plt.show()

def predict(model, num_targets=1):
    test_dataloader_iter = iter(test_dataloader)
    for i in range(0, 1):
        test_batch = next(test_dataloader_iter)
        with torch.no_grad():
            input, target, covariates = utils.split_batch(test_batch)
            input, covariates = utils.scale(input, covariates)
            pred = model(input[:, 0:cond_win_len, :], covariates, future=pred_win_len)
            pred = utils.invert_scale(pred, True)
            loss = model.NLL(pred, target)
            print("loss (prediction): {0}".format(loss))
            preds = pred[0, :, :, :].detach().cpu().numpy()
            targets = target[0, :, :].detach().cpu().numpy()

        for j in range(num_targets):
            pred = preds[:, j, 0]  
            std = preds[:, j, 1]  
            target = targets[:, j]
            plt.figure()
         
            plt.xlabel('x')
            plt.ylabel('y')
            plt.xticks()
            plt.yticks()
            # target series red,condioning window prediction blue,prediciton window prediciton dashesd cyan
            plt.plot(np.arange(len(target)), target, 'r', linewidth=2.0)
            plt.plot(np.arange(cond_win_len), pred[0:cond_win_len], 'b', linewidth=2.0)
            plt.plot(np.arange(cond_win_len, cond_win_len + pred_win_len), pred[cond_win_len:], 'g' + ':',
                     linewidth=2.0)
            # quantile values shades 
            plt.fill_between(np.arange(len(target)), pred - std, pred + std, color='cyan', alpha=0.5)
            plt.fill_between(np.arange(len(target)), pred - 2 * std, pred + 2 * std, color='cyan', alpha=0.2)
            plt.savefig(j)

model, losses = train()

data='elec'# electricity dataset
date_=str(now).replace(" ", "")
model_save_ = f"{data}_epochs{params['num_epochs']}_{ date_ }"
torch.save(model.state_dict(), model_save_+".pth")

predict(model, num_targets)

