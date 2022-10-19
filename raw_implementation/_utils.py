import torch
import numpy as np
 """
 utility functions for  calculation of  error metrics and spliting batches while scaling the batch data
 """

class utils:
    def __init__(self, dev,_params):
        self.min = 0
        self.max = 0
        self.range = 0
        self.mean = 0
        self.dev = dev
        self._params = _params

    def NRMSE(self, input, target):#Normalized Root Mean Square Error
        return np.sqrt(torch.nn.functional.mse_loss(input, target).item()) / torch.mean(torch.abs(target))

    def NormDeviation(self, input, target):#Normalized Deviation
        return torch.mean(torch.abs(input - target)) / torch.mean(torch.abs(target))

    def MAE(self, input, target):#Mean Absolute Error
        return torch.mean(torch.abs(input - target))

    def split_batch(self, batch):
        """
        This was basically done with more of a dependent dataset, right now we are not working with covariates ,
        so each series is considered independent.
        -> functions inputs a batch and divides into input, target and covariates. 
           batch dim-> B x T x D
           D can be defined as :| num_time_indx | num_target_series | num_covariates |
           -  num_time_indx cols: consist of relative and absolute time indices.   
           -  num_target_series cols: number of target series for which we want to perform forecasting
           -  remaining cols: covariates that can be  seasonal (time) covariates (hour, min)
        """
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
        min = torch.min(input, dim=1)[0].unsqueeze(1)
        max = torch.max(input, dim=1)[0].unsqueeze(1)
        self.range = max - min
        self.range[self.range < 1e-5] = 1e-5
        input = input / self.range

        num_covariates = self._params['num_covariates']
        if (num_covariates > 0):
            mean = torch.mean(covariates, dim=1).unsqueeze(1)
            covariates = covariates - mean
            min = torch.min(covariates, dim=1)[0].unsqueeze(1)
            max = torch.max(covariates, dim=1)[0].unsqueeze(1)
            range = max - min
            range[range < 1e-5] = 1e-5
            covariates = covariates / range
        return input, covariates
"""
Inverts the scale applied to the target time series for calculation of  the loss functions on the un-scaled time series.  
"""

    def invert_scale(self, input, probabalistic=False):
        if probabalistic == False:
            return input * self.range + self.mean
        else:
            mean = input[:, :, :, 0]
            std = input[:, :, :, 1]
            scaled_mean = mean * self.range + self.mean
            scaled_std = std * torch.sqrt(self.range)
            return torch.cat((scaled_mean.unsqueeze(-1), scaled_std.unsqueeze(-1)), -1)


