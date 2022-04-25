import torch

from torch.utils.tensorboard import SummaryWriter
from .pytorch_utils import ptu_device, to_numpy
from .utils import create_param_save_path


class FoodTasterTrainer(object):
    def __init__(self, model, train_loader, test_loader, params):
        """
        params:
            configs: opt_args, alpha
            bash: log_dir, param_save_dir, epochs, if_notebook
        """
        self.params = params
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.opt = self.params["opt_args"]["class"](self.model.parameters(), **self.params["opt_args"]["args"])
        self.writer = SummaryWriter(log_dir=self.params["log_dir"])
        self.global_steps = {"train": 0, "eval": 0, "epoch": 0}
        self.model_save_path = create_param_save_path(self.params["param_save_dir"], "food_taster.pt")

    def train_(self):
        """
        tags: train_loss
        """
        self.model.train()
        if self.params["if_notebook"]:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm
        pbar = tqdm(self.train_loader, total=len(self.train_loader), leave=False, desc="training")

        acc = 0
        loss_avg = 0
        for X1, X2, X3 in pbar:
            X1 = X1.float().to(ptu_device)
            X2 = X2.float().to(ptu_device)
            X3 = X3.float().to(ptu_device)
            X1 = self.model(X1)
            X2 = self.model(X2)
            X3 = self.model(X3)

            loss, num_correct_pred = self.compute_loss_and_acc_(X1, X2, X3)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            loss_avg += loss.item() * X1.shape[0]
            acc += num_correct_pred.item() * X1.shape[0]

            # logging
            pbar.set_description(f"loss: {loss.item()}")
            self.writer.add_scalar("train_loss", loss.item(), self.global_steps["train"])
            self.global_steps["train"] += 1

        loss_avg /= len(self.train_loader.dataset)
        acc /= len(self.train_loader.dataset)

        return {"loss": loss_avg, "acc": acc}

    @torch.no_grad()
    def eval_(self):
        """
        tags: eval_loss
        """
        self.model.eval()

        if self.params["if_notebook"]:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm
        pbar = tqdm(self.test_loader, total=len(self.test_loader), leave=False, desc="eval")

        acc = 0
        loss_avg = 0
        for X1, X2, X3 in pbar:
            X1 = X1.float().to(ptu_device)
            X2 = X2.float().to(ptu_device)
            X3 = X3.float().to(ptu_device)
            X1 = self.model(X1)
            X2 = self.model(X2)
            X3 = self.model(X3)

            loss, acc_batch = self.compute_loss_and_acc_(X1, X2, X3)
            loss_avg += loss.item() * X1.shape[0]
            acc += acc_batch.item() * X1.shape[0]

            # logging
            pbar.set_description(f"loss: {loss.item()}")
            self.writer.add_scalar("eval_loss", loss.item(), self.global_steps["eval"])
            self.global_steps["eval"] += 1

        loss_avg /= len(self.test_loader.dataset)
        acc /= len(self.test_loader.dataset)

        return {"loss": loss_avg, "acc": acc}

    def train(self):
        """
        tags: eval_loss
        """
        if self.params["if_notebook"]:
            from tqdm.notebook import trange
        else:
            from tqdm import trange
        pbar = trange(self.params["epochs"], desc="epoch")

        lowest_loss = float("inf")
        for epoch in pbar:
            train_info = self.train_()
            eval_info = self.eval_()

            # save the best_model
            if eval_info["loss"] < lowest_loss:
                lowest_loss = eval_info["loss"]
                torch.save(self.model.state_dict(), self.model_save_path)

            # logging
            desc = f"train: loss: {train_info['loss']:.3f}, acc: {train_info['acc']:.3f}\n" \
                   f"eval: loss: {eval_info['loss']:.3f}, acc: {eval_info['acc']:.3f}"
            pbar.set_description(desc=desc)

            for key in train_info:
                tag = f"epoch_train_{key}"
                self.writer.add_scalar(tag, train_info[key], self.global_steps["epoch"])
            for key in eval_info:
                tag = f"epoch_eval_{key}"
                self.writer.add_scalar(tag, eval_info[key], self.global_steps["epoch"])
            self.global_steps["epoch"] += 1

    def compute_loss_and_acc_(self, X1, X2, X3):
        # X1, X2, X3: (B, N_features) each; all have been sent to ptu.ptu_device
        dist12 = ((X1 - X2) ** 2).sum(dim=1)  # (B,)
        dist13 = ((X1 - X3) ** 2).sum(dim=1)  # (B,)
        loss = dist12 - dist13 + self.params["alpha"]
        loss[loss < 0] = 0
        loss = loss.mean()
        with torch.no_grad():
            acc = (dist12 < dist13).sum() / X1.shape[0]

        return loss, acc
