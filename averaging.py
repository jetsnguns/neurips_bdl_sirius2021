# import numpy as np
# from numpy.random import normal
from math import sqrt
import torch
import torch.nn.utils.convert_parameters as convert
import copy
import eval


def sample_from_swag(theta_swa, diag, D):
    d = len(theta_swa)
    K = D.shape[1]
    z1 = torch.empty(d).normal_(mean=0,std=1)
    z2 = torch.empty(K).normal_(mean=0,std=1)
    if torch.cuda.is_available():
        # print("GPU available!")
        diag = diag.cuda()
        D = D.cuda()
        z1 = z1.cuda()
        z2 = z2.cuda()

    diag = diag ** (1/2)
    theta_res = theta_swa + 1 / sqrt(2) * diag * z1 + 1 / sqrt(2 * K - 2) * z2 @ D
    return theta_res


def average_models(net_fn, test_loader, theta_swa, diag, D, S):
    p_y = 0

    for i in range(S):
        theta_swag = sample_from_swag(theta_swa, diag, D)
        convert.vector_to_parameters(theta_swag, net_fn.parameters())

        test_acc, all_test_probs = eval.evaluate_fn(net_fn, test_loader)

        print(test_acc)
        p_y += 1 / S * all_test_probs
    return p_y
