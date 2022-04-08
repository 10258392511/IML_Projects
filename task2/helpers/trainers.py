import numpy as np
import torch
import torch.nn as nn
import os

from sklearn.metrics import roc_auc_score, r2_score
from torch.utils.tensorboard import SummaryWriter
from .utils import auc_star_loss, r2_loss
from .configs import conf_device, TESTS, VITALS


class SeqModelTrainer(object):
    def __init__(self, model, train_loader, val_loader, params):
        """
        params: **loss_params, **opt_params, **bash_params
            bash_params: epochs, batch_size, test_split_ratio, seed, cls_weight, last_cls_weight,
                         log_dir, model_save_dir, if_notebook
        """
        self.model = model
        self.params = params
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.opt = self.params["constructor"](self.model.parameters(), **self.params["opt_params"])
        self.loss = nn.MSELoss()
        self.global_steps = {"train": 0, "eval": 0, "epoch": 0}
        self.writer = SummaryWriter(log_dir=self.params["log_dir"])

    def train_(self, epoch_pred, epoch_gt):
        if self.params["if_notebook"]:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm
        pbar = tqdm(self.train_loader, total=len(self.train_loader), desc="training", leave=False)

        score_avg = 0
        for X_ages, X_features, y_cls, y_reg in pbar:
            X_ages = X_ages.float().to(conf_device)
            X_features = X_features.float().to(conf_device)
            y_cls = y_cls.long().to(conf_device)
            y_reg = y_reg.float().to(conf_device)
            y_cls_pred, y_reg_pred = self.model(X_ages, X_features)
            loss_cls_array, loss_cls_auc_array, loss_reg_array, loss_reg_r2_array, score = \
                self.compute_loss_(y_cls_pred, y_cls, y_reg_pred, y_reg, epoch_pred, epoch_gt)

            score_avg += score * X_ages.shape[0]
            loss_cls = sum(loss_cls_array)
            loss_reg = sum(loss_reg_array)
            loss = loss_cls * self.params["cls_weight"] + loss_reg

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            # logging
            self.log_info_(loss_cls_array, loss_cls_auc_array, loss_reg_array, loss_reg_r2_array, score, mode="train")
            pbar.set_description(f"score: {score:.3f}")

        score_avg /= len(self.train_loader.dataset)
        pbar.close()

        return score_avg

    @torch.no_grad()
    def eval_(self):
        if self.params["if_notebook"]:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm
        pbar = tqdm(self.val_loader, total=len(self.val_loader), desc="eval", leave=False)

        score_avg = 0
        y_cls_pred_all, y_cls_all = [], []
        for X_ages, X_features, y_cls, y_reg in pbar:
            X_ages = X_ages.float().to(conf_device)
            X_features = X_features.float().to(conf_device)
            y_cls = y_cls.long().to(conf_device)
            y_reg = y_reg.float().to(conf_device)
            y_cls_pred, y_reg_pred = self.model(X_ages, X_features)
            loss_cls_array, loss_cls_auc_array, loss_reg_array, loss_reg_r2_array, score = \
                self.compute_loss_(y_cls_pred, y_cls, y_reg_pred, y_reg, None, None, mode="eval")

            score_avg += score * X_ages.shape[0]
            y_cls_pred_all.append(y_cls_pred)
            y_cls_all.append(y_cls)

            # logging
            self.log_info_(loss_cls_array, loss_cls_auc_array, loss_reg_array, loss_reg_r2_array, score, mode="eval")
            pbar.set_description(f"score: {score:.3f}")

        score_avg /= len(self.val_loader.dataset)
        epoch_pred = torch.cat(y_cls_pred_all, dim=0)
        epoch_gt = torch.cat(y_cls_all, dim=0)
        pbar.close()

        return score_avg, epoch_pred, epoch_gt

    def train(self):
        if self.params["if_notebook"]:
            from tqdm.notebook import trange
        else:
            from tqdm import trange
        pbar = trange(self.params["epochs"], desc="epochs")

        if not os.path.isdir(self.params["model_save_dir"]):
            os.makedirs(self.params["model_save_dir"])
        model_save_path = os.path.join(self.params["model_save_dir"], "model.pt")

        eval_score_avg, epoch_pred, epoch_gt = self.eval_()
        best_score = eval_score_avg
        for epoch in pbar:
            train_score_avg = self.train_(epoch_pred, epoch_gt)
            eval_score_avg, epoch_pred, epoch_gt = self.eval_()
            if best_score < eval_score_avg:
                best_score = eval_score_avg
                torch.save(self.model.state_dict(), model_save_path)

            # logging
            pbar.set_description(f"train score: {train_score_avg:.3f}, eval score: {eval_score_avg:.3f}")
            self.writer.add_scalar("epoch_train", train_score_avg, self.global_steps["epoch"])
            self.writer.add_scalar("epoch_eval", eval_score_avg, self.global_steps["epoch"])
            self.global_steps["epoch"] += 1

    def compute_loss_(self, y_cls_pred, y_cls, y_reg_pred, y_reg, epoch_pred, epoch_gt, mode="train"):
        assert mode in ["train", "eval"]
        loss_cls_list = []
        loss_cls_auc_list = []
        loss_reg_list = []
        loss_reg_r2_list = []
        for i in range(y_cls_pred.shape[1]):
            if mode == "train":
                # loss_cls_list.append(auc_star_loss(y_cls_pred[:, i], y_cls[:, 0, i], epoch_pred[:, i], epoch_gt[:, 0, i],
                #                                    self.params["auc_p"], self.params["auc_gamma"]))
                loss_cls_list.append(self.loss(y_cls_pred[:, i], y_cls[:, 0, i].float()))

            try:
                loss_cls_auc_list.append(roc_auc_score(y_cls[:, 0, i].detach().cpu().numpy(),
                                                       y_cls_pred[:, i].detach().cpu().numpy()))
            except ValueError:
                loss_cls_auc_list.append(0)

        if mode == "train":
            loss_cls_list[-1] *= self.params["last_cls_weight"]

        for i in range(y_reg_pred.shape[1]):
            if mode == "train":
                # loss_reg_list.append(r2_loss(y_reg_pred[:, i], y_reg[:, 0, i]))
                # loss_reg_list.append(((y_reg_pred[:, i] - y_reg[:, 0, i]) ** 2).mean())
                loss_reg_list.append(self.loss(y_reg_pred[:, i], y_reg[:, 0, i]))
            loss_reg_r2_list.append(r2_score(y_reg[:, 0, i].detach().cpu().numpy(),
                                             y_reg_pred[:, i].detach().cpu().numpy()))

        loss_cls_array = loss_cls_list
        loss_cls_auc_array = np.array(loss_cls_auc_list)
        loss_reg_array = loss_reg_list
        loss_reg_r2_array = np.array(loss_reg_r2_list)
        score_cls = np.mean(loss_cls_auc_array[:-1]) + loss_cls_auc_array[-1]
        score_reg = np.mean(np.maximum(0, 0.5 * loss_reg_r2_array) + 0.5)
        score = (score_cls + score_reg) / 3

        return loss_cls_array, loss_cls_auc_array, loss_reg_array, loss_reg_r2_array, score

    def log_info_(self, loss_cls_array, loss_cls_auc_array, loss_reg_array, loss_reg_r2_array, score, mode="train"):
        """
        tags: train_auc_star_score, train_r2_score, train_auc_score, train_r2_gt_score
        """
        assert mode in ["train", "eval"]
        if mode == "train":
            for tag_name, scalar in zip(TESTS, loss_cls_array):
                self.writer.add_scalar(f"{mode}_{tag_name}_auc_loss", scalar, self.global_steps[mode])

            for tag_name, scalar in zip(VITALS, loss_reg_array):
                self.writer.add_scalar(f"{mode}_{tag_name}_r2_loss", scalar, self.global_steps[mode])

        for tag_name, scalar in zip(TESTS, loss_cls_auc_array):
            self.writer.add_scalar(f"{mode}_{tag_name}_auc", scalar, self.global_steps[mode])

        for tag_name, scalar in zip(VITALS, loss_reg_r2_array):
            self.writer.add_scalar(f"{mode}_{tag_name}_r2", scalar, self.global_steps[mode])

        self.writer.add_scalar(f"{mode}_score", score, self.global_steps[mode])
        self.global_steps[mode] += 1
