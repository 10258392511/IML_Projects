import torch
import pickle


with open("info/vocab.pkl", "rb") as rf:
    config_vocab = pickle.load(rf)


config_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

config_split_ratio = 0.9

config_seq_encoder_params = {
    "vocab_size": len(config_vocab["i2c"]),
    "emb_size": 256,
    "num_layers": 6,
    "nhead": 8,
    "dim_feedforward": 512,
    "dropout": 0.1,
    "norm": True
}

config_mlp_params = {
    "num_features": 1000,
    "embed_size": 5,
    "out_size": 256,
    "hidden_dims": [256] * 6,
    "dropout": 0.1
}

config_fusion_params = {
    "emb_size_all": config_seq_encoder_params["emb_size"] + config_mlp_params["out_size"],
    "hidden_dims": [64],
    "dropout": 0.1
}

config_train_params = {
    "opt_class": torch.optim.AdamW,
    "opt_params": {
        "lr": 1e-3,
    },
    "scheduler_class": torch.optim.lr_scheduler.ExponentialLR,
    "scheduler_params": {
        "gamma": 0.5
    },
    "clip_grad_val": 1,
    "save_filename": "model.pt"
}
