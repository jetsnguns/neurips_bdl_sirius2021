import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

import numpy as np
from numpy.random import normal
from numpy.random import multivariate_normal

from math import sqrt

def param_to_vector(parameters):
  res = []
  for param in parameters:
      res.append(param.view(-1))
  res_tensor = torch.cat(res)   
  res_tensor = res_tensor.cpu() 
  return res_tensor.detach().numpy()

def vector_to_parameters(vector, parameters):
  tensor = torch.from_numpy(vector) 
  # Pointer for slicing the vector for each parameter
  pointer = 0
  for param in parameters:
    # The length of the parameter
    num_param = param.numel()
    # Slice the vector, reshape it, and replace the old data of the parameter
    param.data = tensor[pointer:pointer + num_param].view_as(param).data
    # Increment the pointer
    pointer += num_param

def sample_from_swag(theta_swa, diag, D, d, K):
  z1 = normal(size = d)
  z2 = normal(size = K)
  #print(z1)
  #print(z2)
  theta_res = theta_swa + 1 / sqrt(2) * np.dot(diag, z1) + 1 / sqrt(2 * K - 2) * np.dot(D, K)
  return theta_res

def sample_from_swag1(theta_swa, diag, D, d, K):
  mean1 = np.zeros(d)
  mean2 = np.zeros(K)
  cov1 = np.eye(d)
  cov2 = np.eye(K)
  z1 = multivariate_normal(mean1, cov1, (-1, d))
  z2 = multivariate_normal(mean2, cov2, (-1, K))
  print(z1)
  print(z2)
  #theta_res = theta_swa + 1 / sqrt(2) * np.dot(diag, z1) + 1 / sqrt(2 * K - 2) * np.dot(D, K)
  theta_res = 0
  return theta_res

def average_models(net_fn, test_loader, theta_swa, diag, D, d, K, S)
  p_y = 0

  for i in range(S):
    theta_swag = sample_from_swag(theta_swa, diag, D, d, K)
    vector_to_parameters(theta_swag, net_fn.parameters())

    model_state_dict = copy.deepcopy(net_fn.state_dict())
    test_acc, all_test_probs = evaluate_fn(test_loader, model_state_dict)

    print(test_acc)
    p_y += 1 / S * all_test_probs
  return p_y