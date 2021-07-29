import torch


def get_accuracy_fn(net_fn, batch, model_state_dict):
    x, y = batch
    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()
    # get logits
    net_fn.eval()
    with torch.no_grad():
        for name, param in net_fn.named_parameters():
            param.data = model_state_dict[name]
        logits = net_fn(x)
    net_fn.train()
    # get log probs
    log_probs = F.log_softmax(logits, dim=1)
    # get preds
    probs = torch.exp(log_probs)
    preds = torch.argmax(logits, dim=1)
    accuracy = (preds == y).float().mean()
    return accuracy, probs


def evaluate_fn(net_fn, data_loader, model_state_dict):
    sum_accuracy = 0
    all_probs = []
    for x, y in data_loader:
        batch_accuracy, batch_probs = get_accuracy_fn(net_fn, (x, y), model_state_dict)
        sum_accuracy += batch_accuracy.item()
        all_probs.append(batch_probs)
    all_probs = torch.cat(all_probs, dim=0)
    return sum_accuracy / len(data_loader), all_probs
