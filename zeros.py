import numpy as np
from numpy.random import randint
import torch
import torch.nn.utils.convert_parameters as convert
from torch.nn.functional import softmax
from torch.utils.data import DataLoader

from eval import get_accuracy_fn, evaluate_fn
import swag
import averaging


def zeros(net_fn, log_posterior_fn, 
          trainset, valset, testset, 
          M, T=60, batch_size=128,
          c=2, K=10, S=5,
          lr=1e-4, prior_variance=1, momentum=0):
    
    theta_init = convert.parameters_to_vector(net_fn.parameters())

    n = len(theta_init)
    thetas = [theta_init]
    for m in M:
        zero_idx = randint(0, n, m)
        new_theta = theta_init.clone().detach()
        for j in zero_idx:
            new_theta[j] = 0
        thetas.append(new_theta)
        
    N = len(testset)
    n_classes = 10 

    all_test_probs_swag_np = np.zeros((N, n_classes))
    test_acc = []
    
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    for item in thetas:
        convert.vector_to_parameters(item, net_fn.parameters())
        acc, _ = evaluate_fn(net_fn, val_loader)
        test_acc.append(acc)

    print(test_acc)
    test_acc = torch.tensor(test_acc)
    weigth = softmax(test_acc)
    print(weigth)

    for i in range(len(thetas)):
        item = thetas[i]
        w = weigth[i]
        convert.vector_to_parameters(item, net_fn.parameters())
        teta, diag, D = swag.train_SWAG(net_fn, log_posterior_fn, trainset, 
                                    T, batch_size, c,
                                    lr, K,
                                    prior_variance, momentum)
        
        all_test_probs_swag = averaging.average_models(net_fn, test_loader, teta, diag, D, S)
        all_test_probs_swag_np +=  (w * all_test_probs_swag).cpu().numpy()
    return all_test_probs_swag_np