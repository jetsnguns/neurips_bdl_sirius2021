import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

import numpy as np
import copy
import itertools

def param_to_vector(parameters):
  res = []
  for param in parameters:
      res.append(param.view(-1))
  res_tensor = torch.cat(res)   
  res_tensor = res_tensor.cpu() 
  return res_tensor.detach().numpy()

def train_SWAG(net_fn, log_posterior_fn, trainset, T=1000, S=100, c=50, lr=0.001, K=10):
    
  momentum_decay = 0.9
  optimizer = optim.SGD(net_fn.parameters(), lr=lr, momentum=momentum_decay)

  SWAG_loader = itertools.islice(itertools.cycle(DataLoader(trainset, batch_size=S, shuffle=True)), 0, T)

  teta = param_to_vector(net_fn.parameters())
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
      new_teta = param_to_vector(net_fn.parameters())
      teta = (n*teta + new_teta)/(n+1)
      teta_sqr = (n*teta_sqr + new_teta*new_teta)/(n+1)

      if len(D) > K:
        del D[0]
      D.append(new_teta - teta)

  D = np.array(D)
  return teta, teta_sqr, D