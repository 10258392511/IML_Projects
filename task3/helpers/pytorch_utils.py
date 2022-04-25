import torch


ptu_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def to_numpy(tensor):
    return tensor.detach().cpu().numpy()
