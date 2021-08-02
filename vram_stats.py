import torch


def get_vram_usage_str():
    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    msg = "Using device: {}".format(device)

    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        alloc = round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1)
        cach = round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1)
        msg += f'  Allocated: {alloc} GB; Cached: {cach} GB'

        return msg
