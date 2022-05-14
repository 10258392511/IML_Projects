import torch
import torch.nn as nn

from .configs import config_vocab


class SeqEncoder(nn.Module):
    def __init__(self, params):
        """
        params: configs.config_seq_encoder_params
            vocab_size, emb_size, num_layers, nhead, dim_feedforward, dropout, norm
        """
        super(SeqEncoder, self).__init__()
        self.params = params
        self.embed = nn.Embedding(self.params["vocab_size"], self.params["emb_size"])
        encoder_layer = nn.TransformerEncoderLayer(self.params["emb_size"], self.params["nhead"],
                                                   self.params["dim_feedforward"], self.params["dropout"])
        norm = None
        if self.params["norm"]:
            norm = nn.LayerNorm((self.params["emb_size"]))
        self.encoder = nn.TransformerEncoder(encoder_layer, self.params["num_layers"], norm)

    def forward(self, X, src_key_padding_mask):
        # X: (T, B), src_key_padding_mask: (B, T)
        X = self.embed(X)  # (T, B, N_emb)
        X = self.encoder(X, src_key_padding_mask=src_key_padding_mask)  # (T, B, N_emb)

        # (B, N_emb)
        return X[-1, ...]


class MLP(nn.Module):
    def __init__(self, params):
        """
        params: configs.config_mlp_params
            num_features, embed_size, out_size, hidden_dims: list, dropout
        """
        super(MLP, self).__init__()
        self.params = params
        self.embeds = nn.ModuleList([nn.Embedding(2, self.params["embed_size"]) for _ in range(self.params["num_features"])])
        layers = []
        hidden_dims = [self.params["num_features"] * self.params["embed_size"]] + self.params["hidden_dims"]
        for in_features, out_features in zip(hidden_dims, hidden_dims[1:]):
            layers.append(nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features),
                nn.ReLU(),
                nn.Dropout(self.params["dropout"])
            ))
        layers.append(nn.Linear(hidden_dims[-1], self.params["out_size"]))
        self.layers = nn.Sequential(*layers)

    def forward(self, X):
        # X: (B, N_features)
        # (B, num_features * emb_size)
        X_embeds = torch.cat([self.embeds[i](X[:, i]) for i in range(X.shape[-1])], dim=-1)
        X = self.layers(X_embeds)  # (B, N_emb)

        return X


class Fusion(nn.Module):
    def __init__(self, params):
        """
        params: config_fusion_params
            emb_size_all, hidden_dims: list, dropout
        """
        super(Fusion, self).__init__()
        self.params = params
        hidden_dims = [self.params["emb_size_all"]] + self.params["hidden_dims"]
        layers = []
        for in_features, out_features in zip(hidden_dims, hidden_dims[1:]):
            layers.append(nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.ReLU(),
                nn.Dropout(self.params["dropout"])
            ))
        self.layers = nn.Sequential(*layers)
        self.out_layer = nn.Linear(hidden_dims[-1], 1)

    def forward(self, X_smiles_enc, X_feature_enc):
        # X_smiles_enc, X_feature_enc: (B, N_emb_1), (B, N_emb_2)
        X = torch.concat([X_smiles_enc, X_feature_enc], dim=-1)  # (B, emb_size_all)
        X = self.layers(X)
        X = self.out_layer(X).squeeze()  # (B, 1) -> (B,)

        # (B,)
        return X


class EnergyEstimator(nn.Module):
    def __init__(self, params):
        """
        params: seq_encoder_params, mlp_params, fusion_params
        """
        super(EnergyEstimator, self).__init__()
        self.params = params
        self.seq_encoder = SeqEncoder(self.params["seq_encoder_params"])
        self.mlp = MLP(self.params["mlp_params"])
        self.fusion = Fusion(self.params["fusion_params"])

    def forward(self, X_smiles, X_features):
        # X_smiles: (T, B), X_features: (B, N_features)
        src_padding_mask = (X_smiles == config_vocab["c2i"]["<PAD>"]).transpose(0, 1)  # (B, T)
        X_seq_enc = self.seq_encoder(X_smiles, src_padding_mask)
        X_feat_enc = self.mlp(X_features)
        X = self.fusion(X_seq_enc, X_feat_enc)

        # (B,)
        return X
