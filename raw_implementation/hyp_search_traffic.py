# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 01:10:10 2022

@author: user
"""

import os
import numpy as np
import pandas as pd
from pandas import read_csv, DataFrame
from matplotlib import pyplot as plt
from functools import partial
import datetime as dt
# from dataset.electricity_dataset import electricity_pre
from datetime import datetime
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import Sampler, SubsetRandomSampler
import random
import properscoring as ps
from statistics import mean
import gc
import psutil
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

agg_traffic = pd.read_csv('agg_traffic.csv')
agg_mat_max = agg_traffic.values
agg_mat_max = agg_mat_max[:,1:]
AG=agg_mat_max[:7].astype(int)

#  probabalistic model with likelihood function
class Model(nn.Module):
    def __init__(self, num_lstms, input_dim, output_dim, hidden_dim):
        super(Model, self).__init__()
        self.lstm_out = hidden_dim
        self.num_lstms = num_lstms
        lstms = []
        lstms.append(nn.LSTMCell(input_dim, self.lstm_out))
        for i in range(1, self.num_lstms):
            lstms.append(nn.LSTMCell(self.lstm_out, self.lstm_out))
        self.lstms = nn.ModuleList(lstms)
        # μ and σ  distibution -> next point for every t_i.prediction length is from output dimensions
        self.mean = nn.Linear(self.lstm_out, output_dim)
        self.std = nn.Linear(self.lstm_out, output_dim)
        
    def forward(self, input, covariates, future = 0):
        dev = input.device
        means = torch.Tensor().to(dev)
        stds = torch.Tensor().to(dev)
        outputs = []
        h_t = []
        c_t = []
        cond_ctx_len = input.size(1)
        pred_ctx_len = future
        if covariates.shape[2] != 0:
            input = torch.cat((input, covariates[:, 0:cond_ctx_len, :]), 2)
        for i in range(0, self.num_lstms):
            h_t.append(torch.zeros(input.size(0), self.lstm_out, dtype=torch.float).to(dev))
            c_t.append(torch.zeros(input.size(0), self.lstm_out, dtype=torch.float).to(dev))

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            h_t[0], c_t[0] = self.lstms[0](input_t.squeeze(1), (h_t[0], c_t[0]))
            for n in range(1, self.num_lstms):
                h_t[n], c_t[n] = self.lstms[n](h_t[n - 1], (h_t[n], c_t[n]))
            mean = self.mean(h_t[n])
            std = self.std(h_t[n])
            means = torch.cat((means, mean.unsqueeze(1)), 1)
            stds = torch.cat((stds, std.unsqueeze(1)), 1)
        stds = self.softplus(stds)
        for i in range(future):
            output_t = torch.cat((mean, covariates[:, cond_ctx_len + i, :]), 1)
            h_t[0], c_t[0] = self.lstms[0](output_t, (h_t[0], c_t[0]))
            for n in range(1, self.num_lstms):
                h_t[n], c_t[n] = self.lstms[n](h_t[n - 1], (h_t[n], c_t[n]))
            mean = self.mean(h_t[n])
            std = self.std(h_t[n])
            std = self.softplus(std) 
            means = torch.cat((means, mean.unsqueeze(1)), 1)
            stds = torch.cat((stds, std.unsqueeze(1)), 1)
        means = means.unsqueeze(-1)
        stds = stds.unsqueeze(-1)
        outputs = torch.cat((means, stds), -1)
        return outputs

    def sample(self, mean, std):
        normal_dist = torch.distributions.normal.Normal(mean, std)
        return normal_dist.sample()

    def softplus(self, x):
        softplus = torch.log(1+torch.exp(x))
        softplus = torch.where(softplus==float('inf'), x, softplus)
        return softplus

    def NLL(self, outputs, truth,AG,device):#negative log liklihood
        hierarchy_tot,bottom_tot=AG.shape
        mean, std = torch.split(outputs, 1, dim=3)
        AG_tensor= torch.from_numpy(AG).to(device)    
        mean = mean.squeeze(3)
        std = std.squeeze(3)
        #std=std**2
        #bottom value calculation
        bottom_mean= mean[:, :, -bottom_tot:]
        bottom_std=std[:, :, -bottom_tot:]
        #hierarchy value calculation
        Hierarchy_mean=torch.matmul(bottom_mean.float(),AG_tensor.T.float())
        Hierarchy_std=torch.matmul(bottom_std.float()**2,AG_tensor.T.float())**0.5
        #all values for loss calculation top values are hierarchy and lower values are bottom
        total_mean=torch.cat((Hierarchy_mean,bottom_mean),dim=2)
        total_std=torch.cat((Hierarchy_std,bottom_std),dim=2)
        #torch.pi = torch.acos(torch.zeros(1)).item()
        loss = torch.mean((0.5*torch.log(2*torch.pi*(total_std**2)))+torch.div(torch.sub(total_mean,truth)**2, total_std**2))
        return loss
    
    
class _utils:
    def __init__(self, dev,_params):
        self.min = 0
        self.max = 0
        self.range = 0
        self.mean = 0
        self.dev = dev
        self._params = _params

    def split_batch(self, batch):
        num_time_idx = self._params['num_time_idx']
        num_targets = self._params['num_targets']
        total_num_targets = self._params['total_num_targets']
        num_covariates = self._params['num_covariates']
        b = num_time_idx
        time_idx = batch[:, 1::, :b].to(self.dev).float()
        e = b + num_targets
        input = batch[:, 0:-1, b: e].to(self.dev).float()
        target = batch[:, 1::, b: e].to(self.dev).float()
        e = e - num_targets + total_num_targets
        total_num_covariates = batch.shape[2] - e + 1  
        covariates = time_idx
        b = e
        e = b + num_covariates
        if (num_covariates > 0):
            covariates = torch.cat((time_idx, batch[:, 1::, b: e]), -1)  
        return input, target, covariates

    def scale(self, input, covariates):
        self.mean = torch.mean(input, dim=1).unsqueeze(1)
        input = input - self.mean
        self.range = torch.std(input, dim=1).unsqueeze(1)
        input = input / self.range
        return input, covariates
    
    def invert_scale(self, input, probabalistic=False):
        mean = input[:, :, :, 0]
        std = input[:, :, :, 1]
        scaled_mean = mean * self.range + self.mean
        scaled_std = std * torch.sqrt(self.range)
        return torch.cat((scaled_mean.unsqueeze(-1), scaled_std.unsqueeze(-1)), -1)
    
class MyDataset(Dataset):
    def __len__(self):
        return self.dataset_size
    def __getitem__(self, idx):
        rel_time = torch.from_numpy(np.arange(0, self.ctx_win_len) * self.resolution).unsqueeze(1).float().to(self.dev)
        abs_time = torch.from_numpy(np.arange(idx, idx + self.ctx_win_len) * self.resolution).unsqueeze(1).float().to(
            self.dev)
        if (self.num_time_indx == 2):  
            out = torch.cat((rel_time, abs_time, self.scaled[idx: idx + self.ctx_win_len, :]), -1).unsqueeze(0)
        if (self.num_time_indx == 1):  
            out = torch.cat((rel_time, self.scaled[idx: idx + self.ctx_win_len, :]), -1).unsqueeze(0)
        if (self.num_time_indx == 0):  
            out = self.scaled[idx: idx + self.ctx_win_len, :].unsqueeze(0)
        return torch.squeeze(out, 0)
    
    def get_train_test_samplers(self, train_test_split):
        indices = list(range(self.dataset_size))
        split = int(np.floor(train_test_split * self.dataset_size))
        train_indices= indices[: split - self.ctx_win_len]
        val_indices= indices[split: split+self.ctx_win_len]
        test_indices =indices[split+2*self.ctx_win_len:-self.ctx_win_len]
        return train_indices,val_indices,test_indices
    
class Data_pre(MyDataset):
    def __init__(self, csv_file, dev, ctx_win_len, num_time_indx=1):
        self.num_time_indx = num_time_indx
        self.dev = dev
        df = read_csv('/home/jaffery/gluonts-hierarchical-ICML-2021/hyp_param_tuning/trffc.csv',header=0,index_col='date')
        if 'Unnamed: 0' in df.columns.values:
            df.drop('Unnamed: 0', axis=1, inplace=True)
        self.ctx_win_len = ctx_win_len
        values = df.values
        values = values.astype('float32')
        self.scaled = values
        self.dataset_size = self.scaled.shape[0]
        self.scaled = torch.from_numpy(self.scaled).float().to(dev)
        self.resolution = 1
        
_params_ = {
    'num_covariates': 4,
    'total_num_covariates': 2,
    'num_time_idx': 1,
    'num_targets': 207,
    'total_num_targets': 207,
    'data_file': 'trffc.csv',
    'train_test_split':0.7         
}
params=_params_ 

def CRPS_test(out,target,AG,pred_win_len):
    for v in range(out.shape[0]):
        preds = out[0, :, :, :].detach().cpu().numpy()
        targets = target[0, :, :].detach().cpu().numpy()
        preds[:,0:7, 0]=preds[:,7:207, 0]@AG.T
        preds[:,0:7, 1]=((preds[:,7:207, 1])**2@AG.T)**0.5 
    mu,sig=preds[:, :, 0],preds[:, :, 1]
    return(np.mean(ps.crps_gaussian(targets, mu= mu[-pred_win_len:,:], sig=sig[-pred_win_len:,:])))

def train_(config, checkpoint_dir=None):
    global params
    num_covariates = params['num_covariates']
    num_time_idx = params['num_time_idx']
    num_targets = params['num_targets']
    input_dim = num_time_idx + num_targets + num_covariates
    ctx_win_len,cond_win_len=config["context length factor"]
    pred_win_len = ctx_win_len - cond_win_len - 1
    batch_size = config['batch_size']
    net = Model(config['num_lstms'],input_dim=input_dim, output_dim=num_targets, hidden_dim=config['hidden_dim'])
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)    
    net.to(device)
   
    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
    
    dataset_ = Data_pre(csv_file=params['data_file'], dev=device, ctx_win_len=ctx_win_len, num_time_indx=1)
    optimizer = optim.AdamW(net.parameters(), lr=config["lr"],weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["step_size"], gamma=config["gamma"])

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        
    train_, val_,test_ = dataset_.get_train_test_samplers(params['train_test_split'])
    train_sampler,val_sampler,test_sampler=SubsetRandomSampler(train_),SubsetRandomSampler(val_),SubsetRandomSampler(test_)
    train_dataloader = DataLoader(dataset_, batch_size=int(config["batch_size"]), sampler=train_sampler, shuffle=False, num_workers=0)
    val_dataloader= DataLoader(dataset_, batch_size=int(config["batch_size"]), sampler=val_sampler, shuffle=False, num_workers=0)
    test_dataloader = DataLoader(dataset_, batch_size=int(config["batch_size"]), sampler=test_sampler,shuffle=False, num_workers=0)
 
    num_epochs = config["num_epochs"]
    crps=[]
    utils = _utils(device, params)
    for epoch in range(0, num_epochs):
        batch_num = 0
        running_loss = 0.0
        #print(f"epoch:{epoch} /{num_epochs}")
        net.train()
        for i, batch in enumerate(train_dataloader,0):
            input, target, covariates = utils.split_batch(batch) 
            #splitting batches : issue with sequence
            #conditioning window input,Prediction window , model op conditioned on previous values
            # model output for the previous step
            input_cond = input[:, 0:cond_win_len, :]
            input_cond, covariates = utils.scale(input_cond, covariates)
            optimizer.zero_grad()
            #forward pass
            out = net(input_cond, covariates, future=pred_win_len)
            out = utils.invert_scale(out)
            #rescale
            #test_plot(out,target,AG,epoch)
            crps.append(CRPS_test(out,target,AG,pred_win_len))
            loss = net.NLL(out, target,AG,device)
            #loss calc
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss+=loss.item()
            batch_num = batch_num + 1
        #print("Train loss: {0}".format(running_loss/batch_num))
        #print("Avg Crps",mean(crps))
        #print('─' * 30)   
        crps_val=[]
    # Validation loss
        val_loss = 0.0
        val_steps = 0
        net.eval()
        for i, data in enumerate(val_dataloader, 0):
            with torch.no_grad():
                input, target, covariates = utils.split_batch(data)
                input, covariates = utils.scale(input, covariates)
                pred = net(input[:, 0:cond_win_len, :], covariates, future=pred_win_len)
                pred = utils.invert_scale(pred, True)
                loss = net.NLL(pred, target,AG,device)
                #print("Val: {0}".format(loss))     
                val_loss += loss.cpu().numpy()
                val_steps += 1
                crps_val.append(CRPS_test(pred,target,AG,pred_win_len))
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint_")
            torch.save((net.state_dict(), optimizer.state_dict()), path)
        tune.report(loss=(val_loss/val_steps),CRPS=mean(crps_val))
        #gc.set_threshold(80)
    print("Finished Training")
    
    
def test_accuracy(net,AG,batch_s,device="cpu"):
    loss_crps=[]
    train_, val_,test_ = dataset_.get_train_test_samplers(params['train_test_split'])
    test_dataloader = DataLoader(dataset_, batch_size=int(batch_s), sampler=test_sampler,shuffle=False, num_workers=0)
    for i, data in enumerate(test_dataloader, 0):
            with torch.no_grad():
                input, target, covariates = utils.split_batch(data)
                input, covariates = utils.scale(input, covariates)
                pred = net(input[:, 0:cond_win_len, :], covariates, future=pred_win_len)
                pred = utils.invert_scale(pred, True)
                loss = net.NLL(pred, target,AG,device)
                loss_crps.append(CRPS_test(pred,target,AG))
    return mean(loss_crps)

def main(num_samples=1000, max_num_epochs=50, gpus_per_trial=2):
    config = {
    "num_lstms":tune.choice([2,3,4,5,6,8]),
    "hidden_dim": tune.choice([32, 64, 128, 256]),
    "lr": tune.loguniform(1e-5, 1e-2),
    "batch_size": tune.choice([32,64,128,256,512]),
    "context length factor":tune.choice([[10,8],[20,18],[30,28],[50,48]])
    "step_size":tune.choice([10,30,100,1000]),
	"gamma":tune.loguniform(1e-3,1e-1),
	'weight_decay': tune.loguniform(1e-3,1e-1),
	"num_epochs":tune.choice([40,50,60])}
    
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=5,
        reduction_factor=2)
    
    reporter = CLIReporter(
        metric_columns=["loss", "CRPS", "training_iteration"])
    
    result = tune.run(
        partial(train_),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    
    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation CRPS: {}".format(
        best_trial.last_result["CRPS"]))
    
    
if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=900, gpus_per_trial=1)
    
