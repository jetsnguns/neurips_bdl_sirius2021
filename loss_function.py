import copy
import math

import torch
import torch.nn.functional as F


def log_likelihood_fn(net_fn, batch, return_logits=False):
    """Computes the log-likelihood."""
    x, y = batch
    if torch.cuda.is_available():
      x = x.cuda()
      y = y.cuda()
    net_fn.zero_grad()
    logits = net_fn(x)
    num_classes = logits.shape[-1]
    labels = F.one_hot(y.to(torch.int64), num_classes= num_classes)
    softmax_xent = torch.sum(labels * F.log_softmax(logits, dim=1))

    if return_logits:
        return logits, softmax_xent
    else:
        return softmax_xent


def log_prior_fn(net_fn, prior_variance):
    """Computes the Gaussian prior log-density."""
    model_state_dict = copy.deepcopy(net_fn.state_dict())
    n_params = sum(p.numel() for p in model_state_dict.values())
    exp_term = sum((-p**2 / (2 * prior_variance)).sum() for p in model_state_dict.values() )
    norm_constant = -0.5 * n_params * math.log((2 * math.pi * prior_variance))
    return exp_term + norm_constant


def log_posterior_fn(net_fn, batch, prior_variance, return_logits=False):
    log_prior = log_prior_fn(net_fn, prior_variance)

    if return_logits:
        logits, log_lik = log_likelihood_fn(net_fn, batch, return_logits)
        return logits, log_lik + log_prior
    else:
        log_lik = log_likelihood_fn(net_fn, batch)
        return log_lik + log_prior
