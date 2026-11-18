# ReMasker++

**Missingness-aligned masked autoencoding with learned uncertainty for tabular data imputation.**
MA 305 (Data Science) end-term project, Team 2: Jandhyala Mohana Sarath Chandra, Aayush Sharma, Arnav Nitin Birari.

ReMasker++ builds on [ReMasker](https://github.com/alps-lab/remasker) (Du, Melis & Wang, ICLR 2024) and targets
limitations that the ReMasker authors state themselves:

| ReMasker limitation | ReMasker++ component |
|---|---|
| Re-masked cells are uniformly random, unlike real missingness → degrades MCAR → MAR → MNAR | **Missingness-aligned re-masking**: each row is re-masked with the *real* missingness pattern of another row |
| No indication of which imputations can be trusted | **Learned per-cell uncertainty**: a stop-gradient heteroscedastic head predicts each imputation's expected error, without changing the imputations |
| MSE favours RMSE over distributional fidelity (WD) | **Distribution-matching loss** (optional variant, ReMasker++ (W)): exact per-feature 1-D Wasserstein term |
| One-row-at-a-time inference; missing values can leak into the encoder as zeros | **Batched, equivalent architecture** using attention key-padding masks |

Results, the full ablation (including the components that did *not* help) and the report are in
[`report/report.pdf`](report/report.pdf) and [`results/tables/RESULTS.md`](results/tables/RESULTS.md).

## Usage

```python
import numpy as np
from remasker_pp import ReMaskerPP

X = ...                       # numpy array with np.nan for missing values
imp = ReMaskerPP(copy_prob=1.0, var_head=True).fit(X)
X_imputed = imp.transform(X)  # same shape, observed values untouched
sigma = imp.uncertainty_      # predicted error per imputed cell (normalised scale), NaN where observed
```

## Reproducing the experiments

```bash
pip install torch timm hyperimpute xgboost scikit-learn scipy pandas matplotlib tabulate
# main benchmark: 5 datasets x {MCAR, MAR, MNAR} x 3 seeds, ReMasker++ variants + 8 baselines + oracle
python run_grid.py main --methods remasker,pp_base,pp_p,pp_w1,pp_c0.5,pp_c1_v,pp_c1_w0.3_v,pp_m0.3,hyperimpute,missforest,mice,gain,miwae,sinkhorn,softimpute,mean,oracle
# MNAR stress test
python run_grid.py stress --datasets airfoil,compression,raisin --mechs MNAR --rates 0.3,0.5,0.7 --seeds 0,1 \
    --methods pp_c1_v,remasker,hyperimpute,missforest,mean
python aggregate.py main       # -> results/tables, results/figures
```

Variant names: `pp` = batched ReMasker (A); suffixes `_p` pattern embedding, `_c<p>` copy-masking probability,
`_w<λ>` Wasserstein weight, `_v` uncertainty head, `_m<r>` mask ratio. **ReMasker++ = `pp_c1_v`**,
**ReMasker++ (W) = `pp_c1_w0.3_v`**. `+mc` / `+prog` rows are Monte-Carlo / progressive inference from the same model.

## Files

| File | Content |
|---|---|
| `remasker_pp.py` | ReMasker++ model and imputer |
| `official_remasker.py`, `third_party/remasker_official/` | Official ReMasker model code (unmodified apart from an unused import) and a training wrapper |
| `bench.py` | One benchmark job: missingness simulation, imputation, fidelity / downstream / uncertainty metrics |
| `run_grid.py` | Parallel, resumable grid runner (one subprocess per job) |
| `aggregate.py` | Tables and figures |
| `data/` | The five UCI datasets (from the ReMasker repository) |
| `results/raw/` | Raw per-run JSON results |
