import numpy as np
from numpy.random import normal

from math import sqrt
import torch
import torch.nn.utils.convert_parameters as convert
import copy
import eval


def sample_from_swag(theta_swa, diag, D):
    d = len(theta_swa)
    K = D.shape[0]
    z1 = normal(size=d)
    z2 = normal(size=K)
    diag = diag ** (1/2)
    theta_res = theta_swa + 1 / sqrt(2) * diag * z1 + 1 / sqrt(2 * K - 2) * np.dot(z2, D)
    return theta_res


def average_models(net_fn, test_loader, theta_swa, diag, D, S):
    p_y = 0

    for i in range(S):
        theta_swag = sample_from_swag(theta_swa, diag, D)
        tensor = torch.from_numpy(theta_swag).cuda()
        tensor = tensor.float()
        convert.vector_to_parameters(tensor, net_fn.parameters())

        model_state_dict = copy.deepcopy(net_fn.state_dict())
        test_acc, all_test_probs = eval.evaluate_fn(net_fn, test_loader, model_state_dict)

        print(test_acc)
        p_y += 1 / S * all_test_probs
    return p_y
