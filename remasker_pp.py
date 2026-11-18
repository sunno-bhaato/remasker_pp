"""
ReMasker++ : Missingness-Aware Masked Autoencoding with
Uncertainty-Guided Progressive Imputation.

Builds on ReMasker (Du, Melis & Wang, ICLR 2024). Components:

  (A) Batched, padding-mask transformer  - mathematically the same encoder/decoder
      as ReMasker (MAE that only lets the encoder see visible cells), but
      implemented with attention key-padding masks so that rows with different
      numbers of observed values can be processed in one batch. Removes the
      one-row-at-a-time inference of the official code and the "missing value
      fed as 0" corner case of its gather-based masking.
  (B) Missingness-pattern embedding      - the row's *natural* missingness
      indicator vector is embedded and added to the CLS token and decoder
      tokens, so the model can exploit informative missingness (MAR / MNAR).
  (C) Hybrid copy-masking                - each training row is re-masked either
      uniformly at random (ReMasker) or, with prob. `copy_prob`, with the real
      missingness pattern of another row (cf. CACTI, ICML'25), so that the
      re-masking distribution resembles the true missingness distribution.
  (D) Distribution-matching loss         - MSE + lambda * per-feature 1-D
      Wasserstein-2 between predicted and true values of the re-masked cells
      (exact via sorting); counters the variance shrinkage of pure MSE.
  (E) Monte-Carlo re-masking inference   - K stochastic passes that additionally
      hide a random fraction of observed cells -> mean imputation + per-cell
      uncertainty (std).
  (F) Uncertainty-guided progressive imputation - impute in R rounds; in each
      round only the most confident fraction of the remaining missing cells of
      every row is committed and then used as observed context for the rest
      (confidence-ordered iterative decoding, cf. MaskGIT, CVPR'22).
"""

import math
import numpy as np
import torch
import torch.nn as nn

EPS = 1e-8


def sincos_pos_embed(dim: int, length: int) -> torch.Tensor:
    """1-D sin-cos positional embedding with a leading zero row for CLS
    (identical to ReMasker's get_1d_sincos_pos_embed)."""
    omega = np.arange(dim // 2, dtype=float) / (dim / 2.0)
    omega = 1.0 / 10000 ** omega
    out = np.einsum("m,d->md", np.arange(length), omega)
    emb = np.concatenate([np.sin(out), np.cos(out)], axis=1)
    emb = np.concatenate([np.zeros([1, dim]), emb], axis=0)
    return torch.from_numpy(emb).float().unsqueeze(0)  # [1, L+1, D]


class TabMAE(nn.Module):
    def __init__(self, n_feat, embed_dim=32, depth=4, dec_depth=2, heads=4,
                 mlp_ratio=4.0, pattern_embed=False, var_head=False):
        super().__init__()
        D = embed_dim
        self.L = n_feat
        self.embed = nn.Linear(1, D)  # == ReMasker's Conv1d(1, D, k=1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, D))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, D))
        self.register_buffer("pos", sincos_pos_embed(D, n_feat))
        self.register_buffer("dpos", sincos_pos_embed(D, n_feat))

        def layer():
            return nn.TransformerEncoderLayer(
                D, heads, int(D * mlp_ratio), dropout=0.0, activation="gelu",
                batch_first=True, norm_first=True)

        self.enc = nn.TransformerEncoder(layer(), depth, enable_nested_tensor=False)
        self.enc_norm = nn.LayerNorm(D, eps=1e-6)
        self.dec_embed = nn.Linear(D, D)
        self.dec = nn.TransformerEncoder(layer(), dec_depth, enable_nested_tensor=False)
        self.dec_norm = nn.LayerNorm(D, eps=1e-6)
        self.head = nn.Linear(D, 1)
        self.pattern = nn.Linear(n_feat, D) if pattern_embed else None
        # (G) learned uncertainty head on *detached* decoder features: predicts the
        # log of the expected squared error of each cell; cannot alter the imputations.
        self.var_head = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, 1)) if var_head else None
        self.logvar = None
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, x, keep, natmiss):
        """x: [N,L] normalised values (anything at ~keep is ignored)
        keep: [N,L] bool, cells visible to the encoder
        natmiss: [N,L] float, 1 where the value is *naturally* missing"""
        N, L = x.shape
        x = x * keep
        h = self.embed(x.unsqueeze(-1)) + self.pos[:, 1:]
        c = (self.cls_token + self.pos[:, :1]).expand(N, -1, -1)
        pe = None
        if self.pattern is not None:
            pe = self.pattern(natmiss).unsqueeze(1)
            c = c + pe
        h = torch.cat([c, h], 1)
        pad = torch.cat([torch.zeros(N, 1, dtype=torch.bool, device=x.device), ~keep], 1)
        h = self.enc_norm(self.enc(h, src_key_padding_mask=pad))
        d = self.dec_embed(h)
        tok = torch.where(keep.unsqueeze(-1), d[:, 1:], self.mask_token.expand(N, L, -1))
        d = torch.cat([d[:, :1], tok], 1) + self.dpos
        if pe is not None:
            d = d + pe
        d = self.dec_norm(self.dec(d))
        if self.var_head is not None:
            self.logvar = self.var_head(d[:, 1:].detach()).squeeze(-1).clamp(-12, 3)
        return torch.tanh(self.head(d[:, 1:]).squeeze(-1)) / 2 + 0.5


def _random_keep(obs: torch.Tensor, keep_frac: float, gen=None) -> torch.Tensor:
    """Per row keep a random round(keep_frac * n_obs) (>=1) subset of observed cells."""
    r = torch.rand(obs.shape, generator=gen, device=obs.device)
    r[~obs] = 2.0
    n_obs = obs.sum(1)
    n_keep = torch.clamp(torch.round(keep_frac * n_obs), min=1).long()
    n_keep = torch.minimum(n_keep, n_obs)
    rank = r.argsort(1).argsort(1)
    return (rank < n_keep.unsqueeze(1)) & obs


class ReMaskerPP:
    def __init__(self, embed_dim=32, depth=4, dec_depth=2, heads=4, mlp_ratio=4.0,
                 mask_ratio=0.5, batch_size=64, max_epochs=600, max_steps=4000,
                 lr=1e-3, min_lr=1e-5, warmup_frac=40 / 600,
                 pattern_embed=False, copy_prob=0.0, wd_weight=0.0,
                 mc_samples=20, mc_drop=0.25, prog_rounds=4, var_head=False,
                 seed=0, device="cpu", verbose=False):
        self.__dict__.update(locals())
        del self.__dict__["self"]

    # ------------------------------------------------------------------ fit
    def fit(self, X: np.ndarray):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        X = np.asarray(X, dtype=np.float64)
        self.min_ = np.nanmin(X, 0)
        self.max_ = np.nanmax(X, 0)
        Xn = (X - self.min_) / (self.max_ - self.min_ + EPS)
        obs = ~np.isnan(Xn)
        Xt = torch.tensor(np.nan_to_num(Xn), dtype=torch.float32, device=self.device)
        Ot = torch.tensor(obs, device=self.device)
        n, L = Xt.shape

        self.model = TabMAE(L, self.embed_dim, self.depth, self.dec_depth, self.heads,
                            self.mlp_ratio, self.pattern_embed, self.var_head).to(self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, betas=(0.9, 0.95))

        iters = math.ceil(n / self.batch_size)
        epochs = int(min(self.max_epochs, math.ceil(self.max_steps / iters)))
        warm = max(1.0, epochs * self.warmup_frac)
        self.epochs_ = epochs
        self.model.train()
        for ep in range(epochs):
            perm = torch.randperm(n, device=self.device)
            for it in range(iters):
                e = ep + it / iters
                lr = self.lr * e / warm if e < warm else self.min_lr + (self.lr - self.min_lr) * 0.5 * (
                    1 + math.cos(math.pi * (e - warm) / (epochs - warm)))
                for g in opt.param_groups:
                    g["lr"] = lr
                idx = perm[it * self.batch_size:(it + 1) * self.batch_size]
                x, o = Xt[idx], Ot[idx]
                o = o & (o.sum(1, keepdim=True) >= 2)  # rows need >=1 visible and >=1 target
                if o.sum() == 0:
                    continue
                keep = _random_keep(o, 1 - self.mask_ratio)
                if self.copy_prob > 0:
                    nat = ~Ot[idx]
                    donor = nat[torch.randperm(len(idx), device=self.device)]
                    keep_c = o & ~donor
                    valid = (keep_c.sum(1) >= 1) & ((o & donor).sum(1) >= 1)
                    use = (torch.rand(len(idx), device=self.device) < self.copy_prob) & valid
                    keep = torch.where(use.unsqueeze(1), keep_c, keep)
                remask = o & ~keep
                pred = self.model(x, keep, (~Ot[idx]).float())
                se = (pred - x) ** 2
                loss = (se * remask).sum() / remask.sum().clamp(min=1) + \
                       (se * keep).sum() / keep.sum().clamp(min=1)
                if self.var_head:  # Gaussian NLL with detached mean -> calibrated error predictor
                    lv = self.model.logvar
                    nll = lv + (x - pred.detach()) ** 2 * torch.exp(-lv)
                    loss = loss + (nll * remask).sum() / remask.sum().clamp(min=1)
                if self.wd_weight > 0:
                    w, cnt = 0.0, 0
                    for j in range(L):
                        s = remask[:, j]
                        if s.sum() >= 2:
                            a = torch.sort(pred[s, j]).values
                            b = torch.sort(x[s, j]).values
                            w = w + ((a - b) ** 2).mean()
                            cnt += 1
                    if cnt:
                        loss = loss + self.wd_weight * w / cnt
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            if self.verbose and (ep % 100 == 0):
                print(ep, float(loss))
        return self

    # ------------------------------------------------------------ inference
    def _norm(self, X):
        Xn = (np.asarray(X, dtype=np.float64) - self.min_) / (self.max_ - self.min_ + EPS)
        obs = ~np.isnan(Xn)
        return (torch.tensor(np.nan_to_num(Xn), dtype=torch.float32, device=self.device),
                torch.tensor(obs, device=self.device))

    def _denorm(self, Xn):
        return Xn * (self.max_ - self.min_ + EPS) + self.min_

    @torch.no_grad()
    def _predict(self, x, keep, natmiss, chunk=2048):
        out = []
        for i in range(0, len(x), chunk):
            out.append(self.model(x[i:i + chunk], keep[i:i + chunk], natmiss[i:i + chunk]))
        return torch.cat(out)

    @torch.no_grad()
    def _predict_sigma(self, x, keep, natmiss, chunk=2048):
        mu, sd = [], []
        for i in range(0, len(x), chunk):
            mu.append(self.model(x[i:i + chunk], keep[i:i + chunk], natmiss[i:i + chunk]))
            sd.append(torch.exp(0.5 * self.model.logvar))
        return torch.cat(mu), torch.cat(sd)

    @torch.no_grad()
    def _mc(self, x, keep, natmiss, K, gen):
        """deterministic pass + K passes with mc_drop of the visible cells hidden."""
        preds = [self._predict(x, keep, natmiss)]
        for _ in range(K):
            k = _random_keep(keep, 1 - self.mc_drop, gen)
            preds.append(self._predict(x, k, natmiss))
        P = torch.stack(preds)
        return P.mean(0), P[1:].std(0) if K > 1 else torch.zeros_like(P[0])

    @torch.no_grad()
    def transform(self, X, mode="oneshot"):
        """mode: 'oneshot' (ReMasker), 'mc' (E), 'prog' (E+F).
        Returns imputed X in original scale; sets self.uncertainty_ (normalised scale,
        NaN on observed cells)."""
        self.model.eval()
        x, obs = self._norm(X)
        natmiss = (~obs).float()
        gen = torch.Generator(device=self.device).manual_seed(self.seed + 1)
        unc = torch.full_like(x, float("nan"))
        if mode == "oneshot":
            if self.var_head:
                pred, sd = self._predict_sigma(x, obs, natmiss)
                unc = torch.where(obs, unc, sd)
            else:
                pred = self._predict(x, obs, natmiss)
            xf = torch.where(obs, x, pred)
        elif mode == "mc":
            mean, std = self._mc(x, obs, natmiss, self.mc_samples, gen)
            xf = torch.where(obs, x, mean)
            unc = torch.where(obs, unc, std)
        elif mode == "prog":
            xf, filled = x.clone(), obs.clone()
            R = self.prog_rounds
            for r in range(R):
                if self.var_head:  # learned uncertainty: one deterministic pass per round
                    mean, std = self._predict_sigma(xf, filled, natmiss)
                else:
                    mean, std = self._mc(xf, filled, natmiss, self.mc_samples, gen)
                todo = ~filled
                n_todo = todo.sum(1)
                if n_todo.sum() == 0:
                    break
                n_acc = torch.ceil(n_todo.float() / (R - r)).long()  # commit 1/(R-r) of remaining
                s = torch.where(todo, std, torch.full_like(std, float("inf")))
                rank = s.argsort(1).argsort(1)
                acc = todo & (rank < n_acc.unsqueeze(1))
                xf = torch.where(acc, mean, xf)
                unc = torch.where(acc, std, unc)
                filled = filled | acc
        else:
            raise ValueError(mode)
        self.uncertainty_ = unc.cpu().numpy()
        out = self._denorm(xf.cpu().numpy().astype(np.float64))
        Xo = np.asarray(X, dtype=np.float64)
        return np.where(np.isnan(Xo), out, Xo)

    def fit_transform(self, X, mode="prog"):
        return self.fit(X).transform(X, mode)
