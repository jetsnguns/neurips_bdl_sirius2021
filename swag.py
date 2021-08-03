import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import copy
import itertools
import torch.nn.utils.convert_parameters as convert

from tqdm.auto import tqdm


# def param_to_vector(parameters):
#     res = []
#     for param in parameters:
#         res.append(param.view(-1))
#     res_tensor = torch.cat(res)
#     res_tensor = res_tensor.cpu()
#     return res_tensor.detach().numpy()


def train_SWAG(net_fn, log_posterior_fn, trainset,
               T=1000, batch_size=100, c=50, lr=0.001, K=10,
               prior_variance=5.):
    print(type(log_posterior_fn))

    momentum_decay = 0.9
    optimizer = optim.SGD(net_fn.parameters(), lr=lr, momentum=momentum_decay)

    SWAG_loader = itertools.islice(itertools.cycle(DataLoader(trainset, batch_size=batch_size, shuffle=True)), 0, T)

    teta = convert.parameters_to_vector(net_fn.parameters())
    teta_sqr = teta * teta
    teta = teta.cuda()
    teta_sqr = teta_sqr.cuda()
    # D = torch.tensor([])
    D = []
    for i, data in tqdm(enumerate(SWAG_loader), total=T):
        optimizer.zero_grad()

        loss = - log_posterior_fn(net_fn, data, prior_variance)
        loss.backward()
        optimizer.step()

        if (i + 1) % c == 0:
            n = (i + 1) / c
            new_teta = convert.parameters_to_vector(net_fn.parameters())
            teta = (n * teta + new_teta) / (n + 1)
            teta_sqr = (n * teta_sqr + new_teta * new_teta) / (n + 1)

            if len(D) > K:
                del D[0]
            D.append((new_teta - teta).tolist())

    D = torch.tensor(D)
    diag = torch.clamp(teta_sqr - teta * teta, 1.e-30)

    return teta, diag, D
