"""Generate LaTeX tables for the report/slides from results/raw (no hand-copied numbers)."""
import os
import numpy as np
import pandas as pd
from aggregate import load, NAMES, MAIN, ROOT

OUT = os.path.join(ROOT, "report", "tables")
os.makedirs(OUT, exist_ok=True)
df = load("main")
oracle = df[df.method == "oracle"]
df = df[df.method != "oracle"]
MECH = ["MCAR", "MAR", "MNAR"]


def tex_name(m):
    return NAMES[m].replace("ReMasker++", r"\rmpp{}").replace("ReMasker (official)", r"\remasker{} (official)")


def main_table(metrics=("rmse", "wd"), methods=MAIN, std=True):
    d = df[df.method.isin(methods)]
    cells, means = {}, {}
    for met in metrics:
        g = d.groupby(["method", "mech", "seed"])[met].mean().groupby(["method", "mech"]).agg(["mean", "std"])
        for mech in MECH:
            col = g.xs(mech, level="mech")
            best = col["mean"].idxmin() if met in ("rmse", "wd") else col["mean"].idxmax()
            second = col["mean"].drop(best).idxmin() if met in ("rmse", "wd") else col["mean"].drop(best).idxmax()
            for m in methods:
                mu, sd = col.loc[m, "mean"], col.loc[m, "std"]
                s = f"{mu:.4f}" + (rf"{{\scriptsize$\pm${sd:.4f}}}" if std else "")
                if m == best:
                    s = rf"\textbf{{{s}}}"
                elif m == second:
                    s = rf"\underline{{{s}}}"
                cells[(m, met, mech)] = s
    head = " & ".join([rf"\multicolumn{{3}}{{c}}{{{m.upper()} $\downarrow$}}" for m in metrics])
    lines = [r"\begin{tabular}{l" + "ccc" * len(metrics) + "}", r"\toprule",
             rf"Method & {head}\\",
             " & " + " & ".join(MECH * len(metrics)) + r"\\", r"\midrule"]
    for i, m in enumerate(methods):
        row = " & ".join(cells[(m, met, mech)] for met in metrics for mech in MECH)
        lines.append(rf"{tex_name(m)} & {row}\\")
        if i == 2:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def rank_table():
    d = df[df.method.isin(MAIN)]
    R = {}
    for met, asc in [("rmse", True), ("wd", True), ("ds_xgb", False), ("ds_lin", False)]:
        p = d.groupby(["dataset", "mech", "method"])[met].mean().unstack("method")
        R[met] = p.rank(axis=1, ascending=asc).mean()
    R = pd.DataFrame(R)
    R["overall"] = R[["rmse", "wd", "ds_xgb"]].mean(1)
    R = R.sort_values("overall")
    o = oracle.groupby("dataset")[["ds_xgb", "ds_lin"]].mean()
    gap = d.groupby(["method", "dataset"])[["ds_xgb", "ds_lin"]].mean().sub(o, level="dataset").groupby("method").mean()
    lines = [r"\begin{tabular}{lcccc|cc}", r"\toprule",
             r"& \multicolumn{4}{c|}{Average rank (15 settings) $\downarrow$} & \multicolumn{2}{c}{Downstream gap to oracle $\uparrow$}\\",
             r"Method & RMSE & WD & XGB & Overall & XGBoost & Linear\\", r"\midrule"]
    bests = {c: R[c].min() for c in R.columns}
    gb = {c: gap[c].max() for c in gap.columns}
    for m, r in R.iterrows():
        f = lambda c: (rf"\textbf{{{r[c]:.2f}}}" if np.isclose(r[c], bests[c]) else f"{r[c]:.2f}")
        g = lambda c: (rf"\textbf{{{gap.loc[m, c]:+.3f}}}" if np.isclose(gap.loc[m, c], gb[c]) else f"{gap.loc[m, c]:+.3f}")
        lines.append(rf"{tex_name(m)} & {f('rmse')} & {f('wd')} & {f('ds_xgb')} & {f('overall')} & {g('ds_xgb')} & {g('ds_lin')}\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


ABL_ROWS = [("remasker", "\\remasker{} (official)"), ("pp_base", "(A) batched re-implementation"),
            ("pp_p", "(A)+(B) pattern embedding"), ("pp_w1", r"(A)+(D) Wasserstein loss, $\lambda{=}1$"),
            ("pp_base+mc", "(A)+(E) MC re-masking inference"), ("pp_base+prog", "(A)+(E)+(F) progressive inference"),
            ("pp_m0.3", "(A) + uniform mask ratio 0.3 \\emph{(control)}"),
            ("pp_c0.5", "(A)+(C) copy-masking, $p_{\\text{copy}}{=}0.5$"),
            ("pp_c1_v", "\\textbf{\\rmpp}: (A)+(C, $p_{\\text{copy}}{=}1$)+(G)"),
            ("pp_c1_v+mc", "\\rmpp{} + (E) MC inference"), ("pp_c1_v+prog", "\\rmpp{} + (F) progressive"),
            ("pp_c1_w0.3_v", "\\textbf{\\rmpp{}\\,(W)}: \\rmpp{} + (D), $\\lambda{=}0.3$")]


def ablation_table():
    ref = df[df.method == "remasker"].groupby(["dataset", "mech", "seed"])[["rmse", "wd"]].mean()
    lines = [r"\begin{tabular}{lccccc}", r"\toprule",
             r"Variant & RMSE & WD & $\Delta$RMSE & $\Delta$WD & $\Delta$RMSE (MNAR)\\", r"\midrule"]
    for m, lab in ABL_ROWS:
        x = df[df.method == m].groupby(["dataset", "mech", "seed"])[["rmse", "wd"]].mean()
        rel = (x / ref - 1) * 100
        mn = rel.rmse.xs("MNAR", level="mech").mean()
        lines.append(rf"{lab} & {x.rmse.mean():.4f} & {x.wd.mean():.4f} & {rel.rmse.mean():+.1f}\% & {rel.wd.mean():+.1f}\% & {mn:+.1f}\%\\")
        if m in ("remasker", "pp_base+prog", "pp_c0.5"):
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def unc_table():
    rows = [("pp_base+mc", "MC re-masking (E), base model"), ("pp_c1_v+mc", "MC re-masking (E), \\rmpp"),
            ("pp_c1_v", "\\textbf{Learned head (G), \\rmpp}"), ("pp_c1_w0.3_v", "Learned head (G), \\rmpp{}\\,(W)")]
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r"Uncertainty estimate & Spearman $\rho$ $\uparrow$ & RMSE, 25\% most confident & RMSE, 25\% least confident & ratio\\",
             r"\midrule"]
    for m, lab in rows:
        x = df[df.method == m]
        a, b, c = x.unc_spearman.mean(), x.rmse_conf25.mean(), x.rmse_unconf25.mean()
        lines.append(rf"{lab} & {a:.3f} & {b:.4f} & {c:.4f} & {c / b:.2f}$\times$\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def stress_table():
    d = load("stress")
    meths = ["pp_c1_v", "remasker", "hyperimpute", "missforest", "mean"]
    d = d[d.method.isin(meths)]
    t = d.groupby(["method", "rate"])["rmse"].mean().unstack("rate")
    lines = [r"\begin{tabular}{lccc}", r"\toprule", r"Method & 30\% & 50\% & 70\%\\", r"\midrule"]
    for m in meths:
        cells = []
        for r in t.columns:
            v = t.loc[m, r]
            cells.append(rf"\textbf{{{v:.4f}}}" if np.isclose(v, t[r].min()) else f"{v:.4f}")
        lines.append(rf"{tex_name(m)} & " + " & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def per_dataset_table():
    d = df[df.method.isin(MAIN)]
    p = d.groupby(["method", "dataset", "mech"])["rmse"].mean().unstack(["dataset", "mech"])
    ds = ["yacht", "airfoil", "compression", "raisin", "wine"]
    lines = [r"\begin{tabular}{l" + "ccc" * 5 + "}", r"\toprule",
             "Method & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{x}}}" for x in ds) + r"\\",
             " & " + " & ".join(["MC", "MAR", "MN"] * 5) + r"\\", r"\midrule"]
    for m in MAIN:
        cells = []
        for x in ds:
            for mech in MECH:
                v = p.loc[m, (x, mech)]
                cells.append(rf"\textbf{{{v:.3f}}}" if np.isclose(v, p[(x, mech)].min()) else f"{v:.3f}")
        lines.append(rf"{tex_name(m)} & " + " & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


if __name__ == "__main__":
    for name, s in [("main", main_table()), ("main_short", main_table(std=False, methods=MAIN[:6])),
                    ("rank", rank_table()), ("ablation", ablation_table()), ("unc", unc_table()),
                    ("stress", stress_table()), ("per_dataset", per_dataset_table())]:
        open(os.path.join(OUT, f"{name}.tex"), "w").write(s + "\n")
    print(open(os.path.join(OUT, "rank.tex")).read())
