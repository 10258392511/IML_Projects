import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from itertools import chain
from PIL import Image
from typing import List, Tuple
from torch.utils.data import Dataset
from torchvision.models import resnet18
from torchvision.transforms import PILToTensor, Normalize, RandomAffine, ColorJitter, GaussianBlur, \
    RandomHorizontalFlip, RandomVerticalFlip, RandomResizedCrop, Resize
from .utils import convert_txt_to_paths, train_test_split
from .configs import *
from .pytorch_utils import ptu_device, to_numpy


class FoodDataset(Dataset):
    def __init__(self, filename, mode="train"):
        assert mode in ("train", "val", "test"), "invalid mode"
        super(FoodDataset, self).__init__()
        self.mode = mode
        all_paths = convert_txt_to_paths(filename)
        if mode != "test":
            train_paths, test_paths = train_test_split(all_paths)
            self.paths = train_paths if self.mode == "train" else test_paths
        else:
            self.paths = all_paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        assert 0 <= index < len(self.paths), "invalid index"
        paths = self.paths[index]
        imgs = [Image.open(path) for path in paths]
        pil_to_tensor = PILToTensor()
        imgs = [(pil_to_tensor(img) / 255.).to(torch.float32) for img in imgs]
        imgs_out = []

        if self.mode == "train":
            aug_transforms = [
                RandomAffine(**configs_random_affine_param),
                ColorJitter(**configs_color_jitter_param),
                GaussianBlur(**configs_gaussian_blur_param),
                RandomHorizontalFlip(p=1),
                RandomVerticalFlip(p=1)
            ]
            for img in imgs:
                for transform in aug_transforms:
                    sample = np.random.rand()
                    if sample > 0.5:
                        img = transform(img)
                imgs_out.append(img)

            resizer = RandomResizedCrop(**configs_random_crop_resize_param)
            imgs_out = [resizer(img) for img in imgs_out]

        else:
            resizer = Resize(configs_random_crop_resize_param["size"])
            for img in imgs:
                img = resizer(img)
                imgs_out.append(img)

        # resizer = Resize(configs_random_crop_resize_param["size"])
        # for img in imgs:
        #     img = resizer(img)
        #     imgs_out.append(img)
        
        normalizer = Normalize(**configs_normalizer_param)
        imgs_out = [normalizer(img) for img in imgs_out]
        img1, img2, img3 = imgs_out

        return img1, img2, img3

    def get_orig_imgs(self, index):
        assert 0 <= index < len(self.paths), "invalid index"
        paths = self.paths[index]
        imgs = [Image.open(path) for path in paths]
        pil_to_tensor = PILToTensor()
        imgs = [(pil_to_tensor(img) / 255.).to(torch.float32) for img in imgs]

        return imgs

    def visualize_imgs(self, imgs: List[torch.Tensor], **kwargs):
        assert len(imgs) > 0
        mode = kwargs.get("mode", self.mode)
        if mode != "train":
            norm_mean, norm_std = np.array(configs_normalizer_param["mean"]), np.array(configs_normalizer_param["std"])
            de_normalizer = Normalize(-norm_mean / norm_std, 1 / norm_std)
        figsize = kwargs.get("figsize", (3.6 * len(imgs), 4.8))
        fig, axes = plt.subplots(1, len(imgs), figsize=figsize)
        if len(imgs) == 1:
            axes = [axes]

        for axis, img in zip(axes, imgs):
            if mode != "train":
                img = de_normalizer(img)
            img_np = to_numpy(img.permute(1, 2, 0))
            axis.imshow(np.clip(img_np, 0, 1))

        fig.tight_layout()
        plt.show()


class FoodTaster(nn.Module):
    def __init__(self, params):
        """
        params:
            configs: resnet_name, feature_dim
        """
        super(FoodTaster, self).__init__()
        self.params = params
        # self.resnet = torch.hub.load("pytorch/vision:v0.10.0", self.params["resnet_name"], pretrained=True)
        self.resnet = resnet18(pretrained=True)
        self.resnet.eval()
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, self.params["feature_dim"])
        # self.dropout = nn.Dropout(p=0.5)
        # self.head = nn.Sequential(
        #     nn.Linear(self.params["feature_dim"], self.params["feature_dim"]),
        #     nn.ReLU(),
        #     nn.Linear(self.params["feature_dim"], self.params["feature_dim"])
        # )
        self.set_trainable_params()

    def forward(self, X):
        # X: (B, C, H, W)
        X = self.resnet(X)  # (B, N_features)
        # X = self.dropout(X)
        # X = self.head(X)
        X = F.normalize(X, dim=1)

        # (B, N_features)
        return X

    def set_trainable_params(self):
        for param in self.resnet.parameters():
            param.requires_grad = False
        # trainable_modules = [self.resnet.fc, self.dropout, self.head]
        trainable_modules = [self.resnet.fc]
        for module in trainable_modules:
            for param in module.parameters():
                param.requires_grad = True


@torch.no_grad()
def predict(model, imgs: Tuple[torch.Tensor]):
    # imgs: [(B, C, H, W)]
    model.eval()
    imgs = [img.float().to(ptu_device) for img in imgs]
    features1, features2, features3 = [to_numpy(model(img)) for img in imgs]  # (B, N_features) each
    dist12 = np.linalg.norm(features1 - features2, axis=1)
    dist13 = np.linalg.norm(features1 - features3, axis=1)
    pred = (dist12 < dist13).astype(np.int)

    return pred
