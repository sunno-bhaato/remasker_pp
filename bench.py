"""One benchmark job = (dataset, mechanism, miss_rate, seed, method) -> list of result rows.

Fidelity metrics are computed on min-max normalised data (ground-truth scaling),
on the artificially removed cells only:
  RMSE, MAE, WD = mean over features of the 1-D Wasserstein distance between the
  imputed and true values of that feature's missing cells.
Downstream utility: 5-fold CV of XGBoost and a linear model trained on the imputed
features -> AUROC (macro one-vs-rest for multiclass) or R^2.
"""
import json, os, time, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.abspath(__file__))

DATASETS = {  # name -> task
    "yacht": "reg", "airfoil": "reg", "compression": "reg",
    "raisin": "clf", "wine": "clf",
}
NEURAL = ["remasker", "pp_base", "pp_pattern", "pp_copy", "pp_wd", "pp_full"]
BASELINES = ["hyperimpute", "missforest", "mice", "gain", "miwae", "sinkhorn", "softimpute", "mean"]


def load(name):
    df = pd.read_csv(os.path.join(ROOT, "data", f"{name}.csv"), index_col=0)  # index col is NOT a feature
    X = df.iloc[:, :-1].to_numpy(dtype=np.float64)
    y = df.iloc[:, -1].to_numpy()
    return X, y


def ampute(X, mech, rate, seed):
    from hyperimpute.plugins.utils.simulate import simulate_nan
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    mn, mx = X.min(0), X.max(0)
    Xn = (X - mn) / (mx - mn + 1e-8)
    res = simulate_nan(Xn, rate, mech)
    mask = res["mask"].astype(bool)  # True = missing
    Xm = X.copy()
    Xm[mask] = np.nan
    return Xm, mask


def fidelity(X, Ximp, mask):
    from scipy.stats import wasserstein_distance
    mn, mx = X.min(0), X.max(0)
    sc = mx - mn + 1e-8
    T, I = (X - mn) / sc, (Ximp - mn) / sc
    d = (T - I)[mask]
    wds = [wasserstein_distance(T[mask[:, j], j], I[mask[:, j], j]) for j in range(X.shape[1]) if mask[:, j].any()]
    return {"rmse": float(np.sqrt(np.mean(d ** 2))), "mae": float(np.mean(np.abs(d))), "wd": float(np.mean(wds))}


def downstream(Ximp, y, task, seed):
    from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression, Ridge
    from xgboost import XGBClassifier, XGBRegressor
    out = {}
    if task == "clf":
        y = pd.factorize(y, sort=True)[0]
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        score = "roc_auc" if len(np.unique(y)) == 2 else "roc_auc_ovr"
        models = {"xgb": XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, n_jobs=1, verbosity=0),
                  "lin": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))}
    else:
        cv = KFold(5, shuffle=True, random_state=seed)
        score = "r2"
        models = {"xgb": XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, n_jobs=1, verbosity=0),
                  "lin": make_pipeline(StandardScaler(), Ridge())}
    for k, m in models.items():
        out[f"ds_{k}"] = float(np.mean(cross_val_score(m, Ximp, y, cv=cv, scoring=score)))
    return out


def uncertainty_metrics(X, Ximp, mask, unc):
    from scipy.stats import spearmanr
    mn, mx = X.min(0), X.max(0)
    err = np.abs(((X - Ximp) / (mx - mn + 1e-8))[mask])
    u = unc[mask]
    ok = np.isfinite(u)
    err, u = err[ok], u[ok]
    q = np.quantile(u, [0.25, 0.75])
    return {"unc_spearman": float(spearmanr(u, err).correlation),
            "rmse_conf25": float(np.sqrt(np.mean(err[u <= q[0]] ** 2))),  # 25% most confident cells
            "rmse_unconf25": float(np.sqrt(np.mean(err[u >= q[1]] ** 2))),  # 25% least confident cells
            # selective imputation: RMSE of the c most-confident fraction of cells, c = 0.1..1.0
            "cov_curve": [float(np.sqrt(np.mean(err[np.argsort(u)[:max(1, int(c * len(u)))]] ** 2)))
                          for c in np.linspace(0.1, 1.0, 10)]}


ALIASES = {"pp_base": "pp", "pp_pattern": "pp_p", "pp_copy": "pp_c0.5", "pp_wd": "pp_w1",
           "pp_full": "pp_c0.5_w0.3"}


def pp_kwargs(method):
    """pp[_p][_c<copy_prob>][_w<wd_weight>]  e.g. pp_c0.5_w0.3"""
    kw = {}
    for tok in ALIASES.get(method, method).split("_")[1:]:
        if tok == "p":
            kw["pattern_embed"] = True
        elif tok[0] == "c":
            kw["copy_prob"] = float(tok[1:])
        elif tok[0] == "w":
            kw["wd_weight"] = float(tok[1:])
        elif tok == "v":
            kw["var_head"] = True
    return kw


def run_job(dataset, mech, rate, seed, method, max_steps=4000):
    # Import order matters on macOS: loading hyperimpute (-> zmq) / xgboost / sklearn
    # native libs *after* torch segfaults or breaks libomp. Load them first.
    import xgboost, sklearn.linear_model  # noqa: F401
    from hyperimpute.plugins.imputers import Imputers  # noqa: F401
    import torch
    torch.set_num_threads(1)
    task = DATASETS[dataset]
    X, y = load(dataset)
    base = {"dataset": dataset, "mech": mech, "rate": rate, "seed": seed}
    if method == "oracle":
        return [{**base, "method": "oracle", **downstream(X, y, task, seed)}]
    Xm, mask = ampute(X, mech, rate, seed)
    base["actual_rate"] = float(mask.mean())
    rows = []
    t0 = time.time()
    if method == "remasker":
        from official_remasker import OfficialReMasker
        outs = {"remasker": (OfficialReMasker(seed=seed, max_steps=max_steps).fit_transform(Xm), None)}
    elif method.startswith("pp_"):
        from remasker_pp import ReMaskerPP
        m = ReMaskerPP(seed=seed, max_steps=max_steps, **pp_kwargs(method)).fit(Xm)
        xi = m.transform(Xm, "oneshot")
        outs = {method: (xi, m.uncertainty_.copy() if m.var_head else None)}
        if True:  # MC / progressive inference are training-free
            for mode in ("mc", "prog"):
                xi = m.transform(Xm, mode)
                outs[f"{method}+{mode}"] = (xi, m.uncertainty_.copy())
    else:
        from hyperimpute.plugins.imputers import Imputers
        kw = {"random_state": seed} if method not in ("mean",) else {}
        try:
            imp = Imputers().get(method, **kw)
        except Exception:
            imp = Imputers().get(method)
        Xi = np.asarray(imp.fit_transform(pd.DataFrame(Xm)), dtype=np.float64)
        outs = {method: (Xi, None)}
    elapsed = time.time() - t0
    for name, (Xi, unc) in outs.items():
        r = {**base, "method": name, "time": elapsed, **fidelity(X, Xi, mask), **downstream(Xi, y, task, seed)}
        if unc is not None:
            r.update(uncertainty_metrics(X, Xi, mask, unc))
        rows.append(r)
    return rows


if __name__ == "__main__":
    import sys
    d, mech, rate, seed, method = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    rows = run_job(d, mech, rate, seed, method)
    if len(sys.argv) > 6:
        with open(sys.argv[6] + ".tmp", "w") as f:
            json.dump(rows, f)
        os.replace(sys.argv[6] + ".tmp", sys.argv[6])
    for r in rows:
        print(json.dumps(r), flush=True)
