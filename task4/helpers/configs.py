import torch
import numpy as np


config_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

config_split_ratio = 0.9


