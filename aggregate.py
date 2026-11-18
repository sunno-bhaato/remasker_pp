"""Aggregate results/raw/<tag>/*.json into tables (results/tables/*.md, *.csv) and figures (results/figures/*.png)."""
import glob, json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
TAB, FIG = os.path.join(ROOT, "results", "tables"), os.path.join(ROOT, "results", "figures")
os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)

NAMES = {
    "pp_c1_v": "ReMasker++", "pp_c1_w0.3_v": "ReMasker++ (W)", "remasker": "ReMasker (official)",
    "pp_base": "Batched ReMasker (ours, A)", "pp_p": "A + pattern embedding (B)", "pp_c0.5": "A + copy-mask 50% (C)",
    "pp_w1": "A + Wasserstein loss (D)", "pp_m0.3": "A + random mask 30% (control)", "pp_base+mc": "A + MC inference (E)", "pp_base+prog": "A + progressive (E+F)",
    "pp_c1_v+prog": "ReMasker++ + progressive (F)", "pp_c1_v+mc": "ReMasker++ + MC inference (E)",
    "hyperimpute": "HyperImpute", "missforest": "MissForest", "mice": "MICE", "gain": "GAIN", "miwae": "MIWAE",
    "sinkhorn": "Sinkhorn", "softimpute": "SoftImpute", "mean": "Mean",
}
MAIN = ["pp_c1_v", "pp_c1_w0.3_v", "remasker", "hyperimpute", "missforest", "mice", "gain", "miwae", "sinkhorn",
        "softimpute", "mean"]
ABL = ["remasker", "pp_base", "pp_p", "pp_w1", "pp_m0.3", "pp_c0.5", "pp_base+mc", "pp_base+prog", "pp_c1_v", "pp_c1_v+mc",
       "pp_c1_v+prog", "pp_c1_w0.3_v"]
DSORDER = ["yacht", "airfoil", "compression", "raisin", "wine"]

# palette (dataviz reference instance, light mode)
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MUTED = "#b9b8b2"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True})


def load(tag):
    rows = [r for f in glob.glob(os.path.join(ROOT, "results", "raw", tag, "*.json")) for r in json.load(open(f))]
    return pd.DataFrame(rows)


def fmt(m, s):
    return f"{m:.4f} ± {s:.4f}" if np.isfinite(s) else f"{m:.4f}"


def md(df):
    return df.to_markdown()


def main_tables(df, oracle):
    out = []
    d = df[df.method.isin(MAIN)].copy()
    d["name"] = d.method.map(NAMES)
    order = [NAMES[m] for m in MAIN if m in set(d.method)]
    # 1. per-mechanism averages (over datasets; mean of per-dataset seed-means, std over seeds of the dataset-avg)
    for met, asc in [("rmse", True), ("wd", True), ("ds_xgb", False), ("ds_lin", False)]:
        g = d.groupby(["name", "mech", "seed"])[met].mean().groupby(["name", "mech"]).agg(["mean", "std"])
        t = g.apply(lambda r: fmt(r["mean"], r["std"]), axis=1).unstack("mech").reindex(order)[["MCAR", "MAR", "MNAR"]]
        t.to_csv(os.path.join(TAB, f"main_{met}.csv"))
        out.append(f"### {met.upper()} (avg over 5 datasets; mean ± std over 3 seeds; {'lower' if asc else 'higher'} is better)\n\n{md(t)}\n")
    # 2. average rank over (dataset, mech) settings
    ranks = {}
    for met, asc in [("rmse", True), ("wd", True), ("ds_xgb", False)]:
        p = d.groupby(["dataset", "mech", "name"])[met].mean().unstack("name")
        ranks[met] = p.rank(axis=1, ascending=asc).mean().reindex(order)
    R = pd.DataFrame(ranks)
    R["overall"] = R.mean(1)
    R = R.sort_values("overall")
    R.round(2).to_csv(os.path.join(TAB, "avg_rank.csv"))
    out.append("### Average rank across 15 (dataset × mechanism) settings (1 = best)\n\n" + md(R.round(2)) + "\n")
    # 3. wins: settings where method is best
    wins = {}
    for met, asc in [("rmse", True), ("wd", True), ("ds_xgb", False)]:
        p = d.groupby(["dataset", "mech", "name"])[met].mean().unstack("name")
        best = p.idxmin(axis=1) if asc else p.idxmax(axis=1)
        wins[met] = best.value_counts().reindex(order).fillna(0).astype(int)
    W = pd.DataFrame(wins)
    out.append("### Number of settings (out of 15) where each method is best\n\n" + md(W) + "\n")
    # 4. per-dataset RMSE / WD tables
    for met in ["rmse", "wd", "ds_xgb"]:
        p = d.groupby(["name", "dataset", "mech"])[met].mean().unstack(["dataset", "mech"]).reindex(order)
        p = p[[c for ds in DSORDER for c in p.columns if c[0] == ds]]
        p.columns = [f"{a}/{b}" for a, b in p.columns]
        p.round(4).to_csv(os.path.join(TAB, f"per_dataset_{met}.csv"))
    # 5. downstream gap to oracle
    if len(oracle):
        o = oracle.groupby("dataset")[["ds_xgb", "ds_lin"]].mean()
        g = d.groupby(["name", "dataset"])[["ds_xgb", "ds_lin"]].mean()
        gap = g.sub(o, level="dataset").groupby("name").mean().reindex(order)
        gap.columns = ["XGBoost Δ vs oracle", "Linear Δ vs oracle"]
        gap.round(4).to_csv(os.path.join(TAB, "downstream_gap.csv"))
        out.append("### Downstream utility: mean change vs. classifier/regressor trained on complete data "
                   "(AUROC for raisin/wine, R² for yacht/airfoil/compression; closer to 0 is better)\n\n"
                   + md(gap.round(4)) + "\n")
        ot = o.copy(); ot.columns = ["XGBoost (oracle)", "Linear (oracle)"]
        out.append("Oracle scores (complete data):\n\n" + md(ot.round(4)) + "\n")
    return "\n".join(out), R


def ablation_tables(df):
    d = df[df.method.isin(ABL)].copy()
    ref = d[d.method == "remasker"].groupby(["dataset", "mech", "seed"])[["rmse", "wd"]].mean()
    rows = []
    for m in [a for a in ABL if a in set(d.method)]:
        x = d[d.method == m].groupby(["dataset", "mech", "seed"])[["rmse", "wd", "ds_xgb"]].mean()
        rel = (x[["rmse", "wd"]] / ref - 1) * 100
        rows.append({"variant": NAMES[m], "RMSE": x.rmse.mean(), "WD": x.wd.mean(), "XGB": x.ds_xgb.mean(),
                     "ΔRMSE % vs ReMasker": rel.rmse.mean(), "ΔWD % vs ReMasker": rel.wd.mean(),
                     "ΔRMSE MNAR %": rel.rmse.xs("MNAR", level="mech").mean()})
    A = pd.DataFrame(rows).set_index("variant")
    A.round(4).to_csv(os.path.join(TAB, "ablation.csv"))
    return "### Ablation (avg over 5 datasets × 3 mechanisms × 3 seeds; Δ = mean per-setting relative change)\n\n" + md(A.round(4)) + "\n", A


def uncertainty_tables(df):
    d = df[df.unc_spearman.notna()].copy() if "unc_spearman" in df else pd.DataFrame()
    if not len(d):
        return "", None
    d["name"] = d.method.map(NAMES).fillna(d.method)
    U = d.groupby("name")[["unc_spearman", "rmse_conf25", "rmse_unconf25"]].mean()
    U["error ratio (least/most confident)"] = U.rmse_unconf25 / U.rmse_conf25
    U.round(4).to_csv(os.path.join(TAB, "uncertainty.csv"))
    Um = d[d.method == "pp_c1_v"].groupby("mech")[["unc_spearman", "rmse_conf25", "rmse_unconf25"]].mean()
    return ("### Uncertainty quality (Spearman between predicted uncertainty and absolute error on missing cells)\n\n"
            + md(U.round(4)) + "\n\nReMasker++ (learned uncertainty head) by mechanism:\n\n" + md(Um.round(4)) + "\n"), d


# ----------------------------------------------------------------------------- figures
def fig_ablation(A):
    keep = [n for n in A.index if n != NAMES["remasker"]]
    A = A.loc[keep]
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.42 * len(A) + 1.2), sharey=True)
    for ax, col, title in [(axes[0], "ΔRMSE % vs ReMasker", "RMSE change vs official ReMasker (%)"),
                           (axes[1], "ΔWD % vs ReMasker", "WD change vs official ReMasker (%)")]:
        v = A[col]
        hl = [n.startswith("ReMasker++") and "+" not in n.replace("ReMasker++", "") for n in A.index]
        cols = [S[0] if h else MUTED for h in hl]
        y = np.arange(len(A))[::-1]
        ax.barh(y, v, color=cols, height=0.62)
        ax.axvline(0, color=INK2, lw=1)
        for yi, vi in zip(y, v):
            ax.text(vi + (0.6 if vi >= 0 else -0.6), yi, f"{vi:+.1f}%", va="center",
                    ha="left" if vi >= 0 else "right", fontsize=8.5, color=INK)
        ax.set_yticks(y); ax.set_yticklabels(A.index)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.grid(axis="y", visible=False)
        lo, hi = v.min(), v.max()
        ax.set_xlim(min(lo * 1.6, -5), max(hi * 1.35, 5))
    fig.suptitle("Ablation: negative = better than ReMasker", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "ablation.png"), dpi=200)
    plt.close(fig)


def fig_rank(R):
    R = R.sort_values("overall", ascending=False)
    fig, ax = plt.subplots(figsize=(7.5, 0.38 * len(R) + 1.2))
    y = np.arange(len(R))
    mk = {"rmse": ("o", S[0], "RMSE"), "wd": ("s", S[1], "WD"), "ds_xgb": ("D", S[2], "Downstream (XGB)")}
    ax.hlines(y, R[["rmse", "wd", "ds_xgb"]].min(1), R[["rmse", "wd", "ds_xgb"]].max(1), color=GRID, lw=2)
    for c, (m, col, lab) in mk.items():
        ax.scatter(R[c], y, marker=m, s=46, color=col, label=lab, zorder=3, edgecolor=SURF, linewidth=1.5)
    ax.scatter(R["overall"], y, marker="|", s=260, color=INK, label="Overall mean", zorder=4)
    ax.set_yticks(y); ax.set_yticklabels(R.index)
    for t in ax.get_yticklabels():
        if t.get_text().startswith("ReMasker++"):
            t.set_fontweight("bold")
    ax.set_xlabel("Average rank across 15 dataset × mechanism settings (lower is better)")
    ax.invert_xaxis()
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", frameon=False, ncol=4, bbox_to_anchor=(0, 1.0), fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "avg_rank.png"), dpi=200)
    plt.close(fig)


def fig_coverage(dU, ref_rmse=None):
    d = dU[dU.method == "pp_c1_v"]
    if not len(d):
        return
    cov = np.linspace(0.1, 1.0, 10) * 100
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for i, mech in enumerate(["MCAR", "MAR", "MNAR"]):
        c = np.array(d[d.mech == mech].cov_curve.tolist())
        if not len(c):
            continue
        m = c.mean(0)
        ax.plot(cov, m, color=S[i], lw=2, marker="o", ms=4, label=f"ReMasker++, {mech}")
    if ref_rmse is not None:
        ax.axhline(ref_rmse, color=INK2, lw=1.2, ls="--")
        ax.text(2, ref_rmse + 0.002, f"official ReMasker, all cells ({ref_rmse:.3f})", color=INK2, fontsize=8.5)
    ax.set_xlim(0, 103)
    ax.set_xlabel("Fraction of missing cells imputed, most-confident first (%)")
    ax.set_ylabel("RMSE on those cells (normalised)")
    ax.set_title("Selective imputation with ReMasker++'s learned uncertainty", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, loc="lower right")
    ax.set_ylim(0.03, 0.19)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "coverage_curve.png"), dpi=200)
    plt.close(fig)


def fig_downstream(df, oracle):
    if not len(oracle):
        return
    d = df[df.method.isin(MAIN)].copy(); d["name"] = d.method.map(NAMES)
    o = oracle.groupby("dataset")["ds_xgb"].mean()
    g = d.groupby(["name", "dataset"])["ds_xgb"].mean().unstack("dataset")
    gap = (g - o).mean(1).sort_values()
    fig, ax = plt.subplots(figsize=(7, 0.36 * len(gap) + 1.2))
    y = np.arange(len(gap))
    cols = [S[0] if n.startswith("ReMasker++") else MUTED for n in gap.index]
    ax.barh(y, gap.values, color=cols, height=0.62)
    for yi, v in zip(y, gap.values):
        ax.text(v - 0.0015, yi, f"{v:+.3f}", va="center", ha="right", fontsize=8.5, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(gap.index)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_xlim(gap.min() * 1.15, 0.005)
    ax.set_xlabel("XGBoost score − complete-data score\n(AUROC or R², avg over 5 datasets; closer to 0 is better)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "downstream_gap.png"), dpi=200)
    plt.close(fig)


def stress(tag="stress"):
    d = load(tag)
    if not len(d):
        return ""
    d["name"] = d.method.map(NAMES)
    t = d.groupby(["dataset", "name", "rate"])["rmse"].mean().unstack("rate")
    t.round(4).to_csv(os.path.join(TAB, "stress_mnar.csv"))
    dsets = sorted(d.dataset.unique())
    fig, axes = plt.subplots(1, len(dsets), figsize=(4.2 * len(dsets), 3.6), squeeze=False)
    meths = [m for m in ["pp_c1_v", "remasker", "hyperimpute", "missforest", "mean"] if m in set(d.method)]
    for ax, ds in zip(axes[0], dsets):
        for i, m in enumerate(meths):
            s = d[(d.dataset == ds) & (d.method == m)].groupby("rate")["rmse"].mean()
            ax.plot(s.index * 100, s.values, marker="o", ms=4, lw=2, color=S[i], label=NAMES[m])
        ax.set_title(ds, loc="left", fontsize=10, color=INK)
        ax.set_xlabel("MNAR missing rate (%)")
    axes[0][0].set_ylabel("RMSE")
    axes[0][0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "stress_mnar.png"), dpi=200)
    plt.close(fig)
    return "### Stress test: RMSE under MNAR with increasing missing rate (mean over seeds)\n\n" + md(t.round(4)) + "\n"


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "main"
    df = load(tag)
    oracle = df[df.method == "oracle"]
    df = df[df.method != "oracle"]
    print(f"{len(df)} result rows, {df.method.nunique()} methods")
    parts = []
    s, R = main_tables(df, oracle); parts.append(s)
    s, A = ablation_tables(df); parts.append(s)
    s, dU = uncertainty_tables(df); parts.append(s)
    parts.append(stress())
    t = df.groupby("method")["time"].mean().rename(index=NAMES).sort_values()
    parts.append("### Mean wall-clock time per imputation run (s, 1 CPU thread)\n\n" + md(t.round(1).to_frame()) + "\n")
    open(os.path.join(TAB, "RESULTS.md"), "w").write("\n".join(parts))
    fig_ablation(A); fig_rank(R); fig_downstream(df, oracle)
    if dU is not None:
        fig_coverage(dU, df[df.method == "remasker"].rmse.mean())
    print("\n".join(parts))
