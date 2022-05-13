import numpy as np
import pickle

from torch.utils.data import Dataset
from .utils import split_train_set, read_data


data_filename_mapping = {
    "pre-train": ("data/pretrain_features.csv.zip", "data/pretrain_labels.csv.zip"),
    "train": ("data/train_features.csv.zip", "data/train_labels.csv.zip"),
    "test": ("data/test_features.csv.zip", None)
}


class MoleculeDataset(Dataset):
    def __init__(self, mode, seed=0, vocab_path="../info/vocab.pkl"):
        assert mode in ("pre-train", "pre-eval", "train", "eval", "test"), "invalid mode"
        super(MoleculeDataset, self).__init__()
        self.mode = mode
        self.seed = seed
        with open(vocab_path, "rb") as rf:
            self.vocab = pickle.load(rf)  # keys: counter, i2c, c2i
        self.dataset = None  # (ids, SMILEs, features, tgts or None)
        if self.mode == "pre-train":
            self.dataset = split_train_set(*data_filename_mapping["pre-train"], seed=self.seed, mode="train")
        elif self.mode == "pre-eval":
            self.dataset = split_train_set(*data_filename_mapping["pre-train"], seed=self.seed, mode="eval")
        elif self.mode == "train":
            self.dataset = split_train_set(*data_filename_mapping["train"], seed=self.seed, mode="train")
        elif self.mode == "eval":
            self.dataset = split_train_set(*data_filename_mapping["train"], seed=self.seed, mode="eval")
        else:
            self.dataset = read_data(*data_filename_mapping["test"])

        self.max_len = max([len(smile_iter) for smile_iter in self.dataset[1]]) + 2  # plus <BOS> and <EOS>

    def __len__(self):
        return self.dataset[0].shape[0]

    def __getitem__(self, index):
        id = self.dataset[0][index]  # int
        smile_str = self.dataset[1][index]  # str
        feature = self.dataset[2][index]  # (N_feat,)
        tgt = self.dataset[3][index] if self.dataset[3] is not None else None  # float or None

        c2i = self.vocab["c2i"]
        smile_idx = np.array([c2i["<BOS>"]] + [c2i[c_iter] for c_iter in smile_str] + [c2i["<EOS>"]])
        smile_idx_full = np.ones((self.max_len)) * c2i["<PAD>"]
        smile_idx_full[:smile_idx.shape[0]] = smile_idx
        smile_idx_full = smile_idx_full.astype(int)

        return id, smile_idx_full, feature, tgt
