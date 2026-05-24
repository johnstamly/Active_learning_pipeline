# Repo Notes — Active Learning Pipeline for Stepped Composite Repairs

Companion to `README.md` for collaborators (and future Claude sessions) who need a deep, accurate picture of how the code, data, and paper fit together.

Paper: **"Deep Kernel Bayesian Optimization for the Design of Stepped Composite Repairs under Compression"**, submitted to *Composites Part A* (Elsevier), manuscript JCOMA-25-4218. Reviewer comments for Revision 1 live at `Latex/Revision/JCOMA-25-4218-reviews.pdf`.

---

## 1. What the system does (one paragraph)

Given a 12-dimensional repair-patch design vector (`[a_ply1..6, b_ply1..6]` — six elliptical step axes per radius), train a Deep Kernel Gaussian Process surrogate against high-fidelity CZM Abaqus simulations, then run Bayesian active learning (Expected Improvement plus a multi-objective penalty for repair volume) to propose the next FEM run. Two patch categories are handled in parallel: **Category 0** (aligned-orientation, single nesting constraint) and **Category 1** (mixed off-axis plies, with an extra support constraint).

## 2. Repository layout

```
config.yaml                 # single source of truth for paths & hyperparams
config_manager.py           # ConfigManager.get('A.B.C') dot accessor
prepare_data.py             # Excel → processed_data.csv (+ target scaler)
model_definitions.py        # DeepKernel (MLP) + DeepKernelGP (GPyTorch ExactGP)
train_surrogate.py          # train_dkgp_dual_models() — one model per category
find_new_candidate_points.py# Bayesian acquisition: EI / pure_mean / robust
visualize_patches.py        # Plotly 3D staircase + matplotlib 2D top-view
main_orchestrator.py        # 4-step CLI: prepare → train → find → visualize
run_final_inference.py      # find + visualize only, reusing saved artifacts
update_data.py              # after a real FEM run, append the row to the Excel
cleanup_volume.py           # recompute Volume column for existing rows
ablation_study.ipynb        # 5-variant ablation (RMSE, NLL curves)
paper_results_v2.ipynb      # current paper figures (Pareto/L-shape, top-N)
paper_results.ipynb         # superseded
Data_v4.xlsx                # raw FEM dataset (Elips0 + Elips1 sheets)
data/processed/             # processed_data.csv, new_candidate_point_cat_{0,1}.csv
artifacts/                  # dkgp_model_cat_{0,1}.pth, train_data_cat_{0,1}.pt, target_scaler.pkl
Latex/                      # main.tex, main.pdf, figures, Revision/<reviews.pdf>
```

## 3. Data

`Data_v4.xlsx` has two sheets, both 19 columns:
- **Elips0** — 92 rows (Category 0, aligned-orientation patches).
- **Elips1** — 78 rows (Category 1, mixed 0°/90° patches).

Column order (Elips0 canonical):
```
Specimen_ID, a_ply1..a_ply6, b_ply1..b_ply6, Orientation,
Volume, Strength, Predicted_Strength_Mean,
Expected Strength Improvement, Multi Objective Score
```
> **Gotcha**: Elips1 has the column `Expected Strength  Improvement` (double space). `update_data.py` preserves whatever spelling each sheet already uses — do not silently normalize it.

Units: `a_*, b_*` are integer mm in [5, 35]; `Volume` is mm³; `Strength` is kN (ultimate compressive strength). Parent laminate ≈ 75.0 kN, observed recovery asymptote ≈ 71.2 kN (≈ 95 %).

## 4. The pipeline, step by step

### 4.1 `prepare_data.load_and_prepare_data`
- Reads both Excel sheets, concatenates them, adds the binary `Category_C ∈ {0,1}` feature.
- Drops rows with `NaN` in `Strength`.
- **Min-Max scales** the 12 geometric inputs to `[0,1]` using fixed bounds `(A_PLY_MIN, A_PLY_MAX) = (5, 35)` from `config.yaml`.
- **StandardScaler** on the target `Strength`; saves the scaler to `artifacts/target_scaler.pkl`.
- Writes `data/processed/processed_data.csv` (12 scaled features + `Category_C` + scaled target).

### 4.2 `train_surrogate.train_dkgp_dual_models`
- Loads `processed_data.csv`, splits by `Category_C`, then trains one model per category (`train_single_model`).
- Each model:
  - Feature extractor: `DeepKernel(input_dim=12, latent_dim=3, mlp_depth=cfg.MODEL.MLP_DEPTH)`.
  - GP head: `DeepKernelGP` with `ConstantMean` + `ScaleKernel(MaternKernel(nu=2.5))`.
  - Optimizer: `AdamW(lr=1e-3, weight_decay=1e-5)`.
  - Loss: `gpytorch.mlls.ExactMarginalLogLikelihood`.
  - Trained for `MODEL.NUM_EPOCHS = 10000` epochs, logging every 200.
- Persists `artifacts/dkgp_model_cat_{c}.pth` (state_dict) and `artifacts/train_data_cat_{c}.pt` (`{'train_x', 'train_y'}` tensors needed to rebuild the exact GP).

### 4.3 `model_definitions.DeepKernel`
- `dims` start at `input_dim`, decay linearly to `latent_dim` over `mlp_depth` entries:
  - `growth_rate = (latent_dim − input_dim) / max(mlp_depth − 1, 1)`
  - `dims[i] = max(int(input_dim + i·growth_rate), latent_dim // 2)`, and `dims[-1] = latent_dim`.
- With `mlp_depth=6, latent_dim=3` → `[12, 10, 8, 6, 4, 3]` (matches paper Table 3).
- With `mlp_depth=7` (current `config.yaml`) → `[12, 10, 9, 7, 6, 4, 3]` — **does not match the table**.
- Linear → GELU → Dropout(0.1) between hidden layers; last layer outputs latent features without activation.
- **Residual skip**: every `skip_every = 2` layers, the activation `skip_every` steps back is either added (or concatenated if `use_dense=True`) into the current activation, with a `nn.Linear` adapter when dims differ.
- `DeepKernelGP.forward(x)` projects to latent via the MLP, then evaluates the constant mean + Matern 5/2 covariance.

### 4.4 `find_new_candidate_points.find_dual_candidate_points(mode)`
For each category:
1. Load `dkgp_model_cat_{c}.pth` + the matching `train_data`, rebuild `DeepKernel` + `DeepKernelGP`, `eval()`.
2. `f_max = train_y.max()` in scaled space (used by EI).
3. Monte Carlo generate `NUM_RANDOM_SAMPLES` valid integer candidates via `generate_random_valid_point(config, category)`:
   - Sample 6 distinct integers from [5, 35] sorted ascending → `a_plies`.
   - For each ply, draw `b_i ∈ [min_b, min(a_i, 35)]` integer, where `min_b` is `A_PLY_MIN` or `b_{i-1}+1`, and **Cat 1 adds** `min_b = max(min_b, a_{i-1})` (the support constraint).
   - Retry until a feasible sequence is produced.
4. Volume per candidate: `Σ π · a_i · b_i · THICKNESS_CONST`. Normalize linearly into `[0,1]`.
5. Scale candidates with the same `(A_PLY_MIN, A_PLY_MAX)` bounds used for training, predict `μ, σ` from the GP.
6. Compute `raw_strength_score`:
   - `active_learning` → EI: `(μ − f_max)·Φ(Z) + σ·φ(Z)`, `Z = (μ − f_max)/σ`.
   - `pure_mean` → `μ`.
   - `robust` → `μ − 1.96·σ` (LCB, 95 % lower bound).
7. Min-max normalize the strength score, then `final_score = (1 − w)·S_norm − w·V_norm` with `w = VOLUME_WEIGHT = 0.5`. `argmax` selects the winner.
8. Unscale the predicted mean via the saved `target_scaler`; approximate the unscaled σ by `σ_scaled · target_scaler.scale_[0]`.
9. Save `data/processed/new_candidate_point_cat_{c}.csv` with the physical-scale ply values plus `Predicted_Strength_Unscaled, Predicted_Sigma_Unscaled, Predicted_Volume, Strategy_Score_Raw, Multi_Obj_Score, Selection_Mode`.

### 4.5 `visualize_patches.generate_candidate_visualizations`
- Loads each `new_candidate_point_cat_{c}.csv`, replays the ply orientations from `get_run_title` mapping, and emits:
  - A 3D staircase Plotly figure (HTML) of the stepped patch.
  - A 2D top-view matplotlib figure of the stacked ellipses.
- Hard-coded axis limit ±21 mm.

### 4.6 `update_data.update_data_file`
- Backs up `Data_v4.xlsx` to `Data_v4.xlsx.bak`.
- Reads existing sheets, captures their column order, appends the row from `new_candidate_point_cat_{c}.csv` with the **manually entered** new `Strength` value and the model's `Predicted_Strength_Mean`. Recomputes `Volume` using hard-coded thickness 2.0.
- Writes both sheets back to the same file.

### 4.7 `main_orchestrator.main_orchestrator`
CLI: `python main_orchestrator.py --mode {active_learning|pure_mean|robust}` (default `active_learning`).
Sequence: setup directories → STEP 1 prepare → STEP 2 train → STEP 3 find (mode) → STEP 4 visualize → print the selected candidate row.

`PROJECT.CLEANUP_PREVIOUS_RUNS: False` keeps prior artifacts; flip to `True` to wipe `data/processed/` and `artifacts/` first.

### 4.8 `run_final_inference.main`
CLI: `python run_final_inference.py --mode {...}` (default `robust`).
Skips prepare + train; loads existing artifacts; runs find + visualize. Use this when you want to query the trained model with a different acquisition mode without retraining.

## 5. Notebooks

### `paper_results_v2.ipynb` (9 cells, current)
Generates the headline figures: physics-based selection (min volume with `Strength ≥ ~70 kN`), strength/volume efficiency analysis, top-N candidate tables per category, duplicate-design QC. Outputs `final_result_cat{0,1}_physics.png`, `pre_activelearning_result_cat{0,1}_physics.png`, `parameter_evolution_cat{0,1}.png`, `performance_triplet_cat{0,1}.png`, `top_view_Cat{0,1}_RowN.png`.

### `ablation_study.ipynb`
Five-variant ablation with a fixed validation split (Cat 1 by default):
| Name | depth | skip | kernel |
|------|-------|------|--------|
| Final DKGP (Deep+Matern+Skip) | 5 | 2 | matern |
| DKGP (RBF Kernel) | 5 | 2 | rbf |
| Shallow DKGP (2 Layers) | 2 | 0 | matern |
| DKGP w/o Skip Connections | 5 | 0 | matern |
| Standard GP | — | — | matern (with `ard_num_dims`) |

Trains each for 6000 epochs across multiple seeds, tracks RMSE and NLL, and emits `ablation_nll_curve.png` / `ablation_rmse_curve.png` (these PNGs were removed in the latest commit — re-run the notebook to regenerate).

> The notebook contains earlier cells (9, 10, 12) with `depth ∈ {6, 6, 4}` — the depth=5 sweep in cell 13 is the one used for the paper's Table 4.

## 6. Agreed canonical defaults (audit + reconciliation, 2026-05-17)

Settled with the corresponding author on 2026-05-17. These are the ground truth; use them whenever there is a paper-vs-code discrepancy until the revision lands.

| # | Item | Canonical value | Status |
|---|------|----------------|--------|
| 1 | MLP depth / hidden widths | **depth 7, dims `[12,10,9,7,6,4,3]` (6 Linear layers)** — confirmed against saved state_dicts | Paper Table 3 wrong; correct at revision |
| 2 | Two-Stage Phase I/II strategy | **Manually executed**: ~38 (Cat-0) / ~28 (Cat-1) seeds trained shallow+RBF; config edited and pipeline re-run after N≈70 to switch to Deep+Matern | Paper §3.5 word "automatically selected" should change to "switched at the N=70 boundary" |
| 3 | Ply thickness | **2.0 mm** everywhere (search and Excel) | `config.yaml` `THICKNESS_CONSTANT` changed `0.186 → 2.0` on 2026-05-17 |
| 4 | Monte Carlo pool size | **2,000,000** | `config.yaml` `NUM_RANDOM_SAMPLES` changed `1e6 → 2e6` on 2026-05-17 |
| 5 | Acquisition modes | Code keeps three (`active_learning`/`pure_mean`/`robust`); **EI is the canonical published one** | Paper §3.4 mention of the extras is optional |

### Deferred items (overlap reviewer comments — do **not** touch until reviewer-fix pass)

- **ARD claim**: paper says "Matern 5/2 with ARD" but `model_definitions.DeepKernelGP` uses isotropic `MaternKernel(nu=2.5)` without `ard_num_dims`. See R1-2e.
- **"Category 0 and 28" typo** in §3.5 — should read "Category 0 (38 seeds) and Category 1 (28 seeds)". See R1 wording list.

The same defaults table is stored as Claude memory under `project-code-paper-discrepancies`. Update **both** when something changes.

## 7. How to run

```bash
# Full run (prepare + train + find + visualize)
python main_orchestrator.py --mode active_learning

# Re-run only inference with a different acquisition strategy
python run_final_inference.py --mode robust   # LCB
python run_final_inference.py --mode pure_mean # μ only

# After a real FEM result is back, edit Strength in the candidate CSV, then:
python update_data.py        # appends the new row into Data_v4.xlsx (.bak created)

# Notebooks
jupyter lab paper_results_v2.ipynb   # paper figures
jupyter lab ablation_study.ipynb     # Table 4 ablation
```

Dependencies (no `requirements.txt` in repo yet — derive from imports):
`torch`, `gpytorch`, `pandas`, `numpy`, `scipy`, `scikit-learn`, `joblib`, `pyyaml`, `matplotlib`, `seaborn`, `plotly`, `openpyxl`.

## 8. Revision punch list (mirrors reviewer PDF)

See `Latex/Revision/JCOMA-25-4218-reviews.pdf` for the original. High-impact items:

**Reviewer 1**
- R1-1: Add FE modelling detail (plies modelled explicitly in parent + repair?).
- R1-2: Formalise §3.1 (define `h(x; w)`, `φ`, `W`, `z`, `z_i`, `α(x)`, `y_best⁺`).
- R1-2d: Justify how C² Matern captures "sharp drops" (or soften the claim).
- R1-2e: Explain ARD (or remove from text — see discrepancy #2).
- R1-3: Rewrite §4.3 "rapid cooling" heatmap discussion.
- Wording cleanup: "the agent", "intelligent methods", LHS framing, "smoothness prior", MLP ≠ DNN, "Deep Kernel agent", "uncertainty signal", "inverse design problem".
- R1: state typical UCS so Table 4 RMSE values are interpretable; define NLL; define "run"; "efficiency frontier" vs "Pareto front".

**Reviewer 2**
- R2-1: Add an illustration for hard constraint (3) (`a_i ≤ b_{i+1}`).
- R2-2: Fix Table 1 cohesive stiffness units (100 kN/mm³, 35 kN/mm³); use N/mm for cohesive fracture energy for consistency.
- R2-3: Define Cat 0 / Cat 1 inline in §2.1; fix "Category 38" / "Category 0 and 28" typo; clarify "standard optimisation" vs "complex geometry" in Tables 5/6.
- R2-4: Add a data-flow / training pipeline diagram tying §3.1–§3.5 together. (`README.md` and §3 of this file are the closest existing material to base it on.)
- R2-5: Define each "efficiency" variant on first use.
- R2-6: Discuss mesh-dependence of the 95 % strength saturation.
- R2-7: Re-label Figs 6 & 7 ("exploration phase" instead of "initial data"; clarify "parameter values" on the right y-axis).

---

_Last reviewed: 2026-05-17 (immediately after Revision 1 was received)._
