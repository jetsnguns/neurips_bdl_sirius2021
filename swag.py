import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

import numpy as np
import copy

def create_param_array(net_fn):
  res = np.array([])
  for param_set in net_fn.parameters():
    for items in param_set:
      #print(nums.shape)
      items = items.cpu()
      nums = items.detach().numpy()
      res = np.append(res, nums)
  return res

def train_SWAG(net_fn, log_posterior_fn, SWAG_loader, c=50, lr=0.001):
    
  momentum_decay = 0.9
  optimizer = optim.SGD(net_fn.parameters(), lr=lr, momentum=momentum_decay)

  # len(SWAG_loader) = T, batch_size = S

  teta = create_param_array(net_fn)
  teta_sqr = teta*teta

  for i, data in enumerate(SWAG_loader): 
      optimizer.zero_grad()
      model_state_dict = copy.deepcopy(net_fn.state_dict()) 

      loss = - log_posterior_fn(model_state_dict, data)
      loss.backward()
      optimizer.step()

      if i % c == 0:
        n = i/c + 1
        new_teta = create_param_array(net_fn)
        teta = (n*teta + new_teta)/(n+1)
        teta_sqr = (n*teta_sqr + new_teta*new_teta)/(n+1)

  return teta, teta_sqr