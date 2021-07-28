import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

import numpy as np
import copy
import itertools

def create_param_array(net_fn):
  res = np.array([])
  for param_set in net_fn.parameters():
    for items in param_set:
      #print(nums.shape)
      items = items.cpu()
      nums = items.detach().numpy()
      res = np.append(res, nums)
  return res

def train_SWAG(net_fn, log_posterior_fn, trainset, T=1000, S=100, c=50, lr=0.001, K=10):
    
  momentum_decay = 0.9
  optimizer = optim.SGD(net_fn.parameters(), lr=lr, momentum=momentum_decay)

  SWAG_loader = itertools.islice(itertools.cycle(DataLoader(trainset, batch_size=S, shuffle=True)), 0, T)

  teta = create_param_array(net_fn)
  teta_sqr = teta*teta
  D = []

  for i, data in enumerate(SWAG_loader): 
    optimizer.zero_grad()
    model_state_dict = copy.deepcopy(net_fn.state_dict()) 

    loss = - log_posterior_fn(model_state_dict, data)
    loss.backward()
    optimizer.step()

    if (i+1)%c == 0:
      n = i/c
      new_teta = create_param_array(net_fn)
      teta = (n*teta + new_teta)/(n+1)
      teta_sqr = (n*teta_sqr + new_teta*new_teta)/(n+1)

      if len(D) > K:
        del D[0]
      D.append(new_teta - teta)

  D = np.array(D)
  return teta, teta_sqr, D