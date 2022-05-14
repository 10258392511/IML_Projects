import os
import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
from .utils import create_param_save_path
from .configs import config_device


class Trainer(object):
    def __init__(self, model, train_loader, eval_loader, params):
        """
        params:
        bash:
            batch_size, epochs, log_dir, param_save_dir, if_notebook, seed, if_pre_train
        configs.config_train_params:
            opt_class, opt_params, scheduler_class, scheduler_class_params, clip_grad_val, save_filename
        """
        self.params = params
        self.params["if_pre_train"] = self.params.get("if_pre_train", True)
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        if self.params["if_pre_train"]:
            self.opt = self.params["opt_class"](self.model.parameters(), **self.params["opt_params"])
        else:
            self.opt = self.params["opt_class"](self.model.fusion.out_layer.parameters(), **self.params["opt_params"])
        self.scheduler = self.params["scheduler_class"](self.opt, **self.params["scheduler_params"])
        self.model_save_path = create_param_save_path(self.params["param_save_dir"], self.params["save_filename"])
        self.writer = SummaryWriter(self.params["log_dir"])
        self.global_steps = {"train": 0, "epoch": 0}
        self.loss = torch.nn.MSELoss()

    def train_(self):
        if self.params["if_pre_train"]:
            self.model.train()
        else:
            self.model.eval()
            self.model.fusion.out_layer.train()

        if self.params["if_notebook"]:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm
        pbar = tqdm(self.train_loader, total=len(self.train_loader), desc="training", leave=False)

        avg_loss = 0
        for _, X_smiles, X_features, y in pbar:
            X_smiles = X_smiles.transpose(0, 1).to(config_device)  # (B, T) -> (T, B)
            X_features = X_features.to(config_device)  # (B, N_features)
            y = y.to(config_device).float()  # (B,)
            y_pred = self.model(X_smiles, X_features)  # (B,)
            loss = self.loss(y_pred, y)

            self.opt.zero_grad()
            loss.backward()
            if self.params["clip_grad_val"] is not None:
                nn.utils.clip_grad_value_(self.model.parameters(), self.params["clip_grad_val"])
            self.opt.step()

            # logging
            avg_loss += loss.item() * y.shape[0]
            pbar.set_description(f"train loss: {loss: .3f}")
            self.writer.add_scalar("train_loss", loss.item(), self.global_steps["train"])
            self.global_steps["train"] += 1

        pbar.close()

        return avg_loss / len(self.train_loader.dataset)

    @torch.no_grad()
    def eval_(self):
        self.model.eval()
        if self.params["if_notebook"]:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm
        pbar = tqdm(self.eval_loader, total=len(self.eval_loader), desc="eval", leave=False)

        avg_loss = 0
        for _, X_smiles, X_features, y in pbar:
            X_smiles = X_smiles.transpose(0, 1).to(config_device)  # (B, T) -> (T, B)
            X_features = X_features.to(config_device)  # (B, N_features)
            y = y.to(config_device).float()  # (B,)
            y_pred = self.model(X_smiles, X_features)  # (B,)
            loss = self.loss(y_pred, y)

            # logging
            avg_loss += loss.item() * y.shape[0]
            pbar.set_description(f"eval loss: {loss: .3f}")

        pbar.close()

        return avg_loss / len(self.eval_loader.dataset)

    def train(self):
        if self.params["if_notebook"]:
            from tqdm.notebook import trange
        else:
            from tqdm import trange
        pbar = trange(self.params["epochs"], desc="epoch")

        lowest_eval_loss = float("inf")
        for epoch in pbar:
            train_loss = self.train_()
            eval_loss = self.eval_()

            # saving the best model
            if eval_loss < lowest_eval_loss:
                lowest_eval_loss = eval_loss
                torch.save(self.model.state_dict(), self.model_save_path)
            self.scheduler.step()

            # logging
            self.writer.add_scalar("epoch_train_loss", train_loss, self.global_steps["epoch"])
            self.writer.add_scalar("epoch_eval_loss", eval_loss, self.global_steps["epoch"])
            self.global_steps["epoch"] += 1
            lr = None
            for param_group in self.opt.param_groups:
                lr = param_group["lr"]
                break
            pbar.set_description(f"loss: train: {train_loss: .3f}, eval: {eval_loss: .3f}, lr: {lr}")
