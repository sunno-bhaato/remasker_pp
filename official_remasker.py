"""Thin wrapper around the *official* ReMasker model (third_party/remasker_official,
copied verbatim from github.com/alps-lab/remasker; only an unused tkinter import removed).
Training loop follows remasker_impute.py; AMP/GradScaler dropped (CPU), and the
number of epochs is capped by the same optimiser-step budget used for ReMasker++."""
import math, os, sys
from functools import partial
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party", "remasker_official"))
import model_mae  # noqa: E402
from utils import adjust_learning_rate  # noqa: E402

EPS = 1e-8


class OfficialReMasker:
    def __init__(self, embed_dim=32, depth=4, dec_depth=2, heads=4, mlp_ratio=4.0, mask_ratio=0.5,
                 batch_size=64, max_epochs=600, max_steps=4000, lr=1e-3, min_lr=1e-5,
                 warmup_frac=40 / 600, seed=0):
        self.__dict__.update(locals())
        del self.__dict__["self"]

    def fit_transform(self, X):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        X = np.asarray(X, dtype=np.float64)
        mn, mx = np.nanmin(X, 0), np.nanmax(X, 0)
        Xn = (X - mn) / (mx - mn + EPS)
        M = torch.tensor(~np.isnan(Xn), dtype=torch.float32)
        Xt = torch.tensor(np.nan_to_num(Xn), dtype=torch.float32)
        n, L = Xt.shape
        model = model_mae.MaskedAutoencoder(
            rec_len=L, embed_dim=self.embed_dim, depth=self.depth, num_heads=self.heads,
            decoder_embed_dim=self.embed_dim, decoder_depth=self.dec_depth,
            decoder_num_heads=self.heads, mlp_ratio=self.mlp_ratio,
            norm_layer=partial(nn.LayerNorm, eps=EPS), encode_func="linear")
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr, betas=(0.9, 0.95))
        iters = math.ceil(n / self.batch_size)
        epochs = int(min(self.max_epochs, math.ceil(self.max_steps / iters)))
        warm = max(1, round(epochs * self.warmup_frac))
        model.train()
        for ep in range(epochs):
            perm = torch.randperm(n)
            for it in range(iters):
                adjust_learning_rate(opt, it / iters + ep, self.lr, self.min_lr, epochs, warm)
                idx = perm[it * self.batch_size:(it + 1) * self.batch_size]
                loss, _, _, _ = model(Xt[idx].unsqueeze(1), M[idx], mask_ratio=self.mask_ratio)
                if not math.isfinite(loss.item()):
                    continue
                opt.zero_grad()
                loss.backward()
                opt.step()
        model.eval()
        preds = []
        with torch.no_grad():  # official inference: one row at a time
            for i in range(n):
                _, p, _, _ = model(Xt[i].reshape(1, 1, -1), M[i].reshape(1, -1))
                preds.append(p.squeeze(-1))
        P = torch.cat(preds).numpy().astype(np.float64)
        out = P * (mx - mn + EPS) + mn
        return np.where(np.isnan(X), out, X)
