# Repo Notes — Active Learning Pipeline for Stepped Composite Repairs

Companion to `README.md` for collaborators who need a deeper, accurate picture
of how the code, data, and paper fit together.

Paper: **"Deep Kernel Bayesian Optimization for the Design of Stepped Composite
Repairs under Compression"** (G. Stamatelatos et al.), Composites Part A. The
manuscript PDF lives at `paper/manuscript.pdf`.

---

## 1. What the system does (one paragraph)

Given a 12-dimensional repair-patch design vector
(`[a_ply1..6, b_ply1..6]` — six elliptical step axes per radius), train a Deep
Kernel Gaussian Process surrogate against high-fidelity CZM Abaqus simulations,
then run Bayesian active learning (Expected Improvement plus a multi-objective
penalty for repair volume) to propose the next FEM run. Two patch categories
are handled in parallel: **Category 0** (aligned-orientation, single nesting
constraint) and **Category 1** (mixed off-axis plies, with an extra accessibility
constraint imposed by top-down laser scarfing).

## 2. Repository layout

```
config.yaml                   # single source of truth for paths & hyperparams
config_manager.py             # ConfigManager.get('A.B.C') dot accessor
prepare_data.py               # Excel -> processed_data.csv (+ target scaler)
model_definitions.py          # DeepKernel (MLP) + DeepKernelGP (GPyTorch ExactGP)
train_surrogate.py            # train_dkgp_dual_models() - one model per category
find_new_candidate_points.py  # Bayesian acquisition: EI / pure_mean / robust
visualize_patches.py          # Plotly 3D staircase + matplotlib 2D top-view
main_orchestrator.py          # 4-step CLI: prepare -> train -> find -> visualize
run_final_inference.py        # find + visualize only, reusing saved artifacts
update_data.py                # after a real FEM run, append the row to the Excel
cleanup_volume.py             # recompute Volume column for existing rows
ablation_study.ipynb          # 5-variant ablation (RMSE, NLL curves)
paper_results.ipynb           # current paper figures (Pareto/L-shape, top-N)
Data_v4.xlsx                  # raw FEM dataset (Elips0 + Elips1 sheets)
data/processed/               # processed_data.csv, new_candidate_point_cat_{0,1}.csv
artifacts/                    # dkgp_model_cat_{0,1}.pth, train_data_cat_{0,1}.pt, target_scaler.pkl
paper/manuscript.pdf          # published manuscript
```

## 3. Data

`Data_v4.xlsx` has two sheets, both 19 columns:
- **Elips0** — Category 0, aligned-orientation patches.
- **Elips1** — Category 1, mixed 0°/90° patches.

Column order (Elips0 canonical):
```
Specimen_ID, a_ply1..a_ply6, b_ply1..b_ply6, Orientation,
Volume, Strength, Predicted_Strength_Mean,
Expected Strength Improvement, Multi Objective Score
```
> **Note**: Elips1 uses the column name `Expected Strength  Improvement` (with
> a double space). `update_data.py` preserves whatever spelling each sheet
> already uses — do not silently normalize it.

Units: `a_*, b_*` are integer mm in [5, 35]; `Volume` is mm³; `Strength` is
kN (ultimate compressive strength). Parent laminate ≈ 75.0 kN; observed
recovery asymptote ≈ 71.2 kN (≈ 95 %).

## 4. The pipeline, step by step

### 4.1 `prepare_data.load_and_prepare_data`
- Reads both Excel sheets, concatenates them, adds the binary
  `Category_C ∈ {0,1}` feature.
- Drops rows with `NaN` in `Strength`.
- **Min-Max scales** the 12 geometric inputs to `[0,1]` using fixed bounds
  `(A_PLY_MIN, A_PLY_MAX) = (5, 35)` from `config.yaml`.
- **StandardScaler** on the target `Strength`; saves the scaler to
  `artifacts/target_scaler.pkl`.
- Writes `data/processed/processed_data.csv` (12 scaled features +
  `Category_C` + scaled target).

### 4.2 `train_surrogate.train_dkgp_dual_models`
- Loads `processed_data.csv`, splits by `Category_C`, then trains one model
  per category (`train_single_model`).
- Each model:
  - Feature extractor:
    `DeepKernel(input_dim=12, latent_dim=3, mlp_depth=cfg.MODEL.MLP_DEPTH)`.
  - GP head: `DeepKernelGP` with `ConstantMean` +
    `ScaleKernel(MaternKernel(nu=2.5))` (isotropic).
  - Optimizer: `AdamW(lr=1e-3, weight_decay=1e-5)`.
  - Loss: `gpytorch.mlls.ExactMarginalLogLikelihood`.
  - Trained for `MODEL.NUM_EPOCHS = 10000` epochs, logging every 200.
- Persists `artifacts/dkgp_model_cat_{c}.pth` (state_dict) and
  `artifacts/train_data_cat_{c}.pt` (`{'train_x', 'train_y'}` tensors needed
  to rebuild the exact GP).

### 4.3 `model_definitions.DeepKernel`
- `dims` start at `input_dim`, decay linearly to `latent_dim` over `mlp_depth`
  entries:
  - `growth_rate = (latent_dim − input_dim) / max(mlp_depth − 1, 1)`
  - `dims[i] = max(int(input_dim + i·growth_rate), latent_dim // 2)`,
    and `dims[-1] = latent_dim`.
- With `mlp_depth=7, latent_dim=3` (current `config.yaml`) →
  `[12, 10, 9, 7, 6, 4, 3]` (6 Linear layers, 3 residual skip connections).
- Linear → GELU → Dropout(0.1) between hidden layers; last layer outputs
  latent features without activation.
- **Residual skip**: every `skip_every = 2` layers, the activation
  `skip_every` steps back is either added (or concatenated if
  `use_dense=True`) into the current activation, with a `nn.Linear` adapter
  when dims differ.
- `DeepKernelGP.forward(x)` projects to latent via the MLP, then evaluates
  the constant mean + Matérn 5/2 covariance.

### 4.4 `find_new_candidate_points.find_dual_candidate_points(mode)`
For each category:
1. Load `dkgp_model_cat_{c}.pth` + the matching `train_data`, rebuild
   `DeepKernel` + `DeepKernelGP`, `eval()`.
2. `f_max = train_y.max()` in scaled space (used by EI).
3. Monte Carlo generate `NUM_RANDOM_SAMPLES = 2,000,000` valid integer
   candidates via `generate_random_valid_point(config, category)`:
   - Sample 6 distinct integers from [5, 35] sorted ascending → `a_plies`.
   - For each ply, draw `b_i ∈ [min_b, min(a_i, 35)]` integer, where `min_b`
     is `A_PLY_MIN` or `b_{i-1}+1`, and **Cat 1 adds**
     `min_b = max(min_b, a_{i-1})` (the accessibility constraint).
   - Retry until a feasible sequence is produced.
4. Volume per candidate: `Σ π · a_i · b_i · THICKNESS_CONSTANT`. Normalize
   linearly into `[0,1]`.
5. Scale candidates with the same `(A_PLY_MIN, A_PLY_MAX)` bounds used for
   training, predict `μ, σ` from the GP.
6. Compute `raw_strength_score`:
   - `active_learning` → EI: `(μ − f_max)·Φ(Z) + σ·φ(Z)`,
     `Z = (μ − f_max)/σ`.
   - `pure_mean` → `μ`.
   - `robust` → `μ − 1.96·σ` (LCB, 95 % lower bound).
7. Min-max normalize the strength score, then
   `final_score = (1 − w)·S_norm − w·V_norm` with `w = VOLUME_WEIGHT = 0.5`.
   `argmax` selects the winner.
8. Unscale the predicted mean via the saved `target_scaler`; approximate the
   unscaled σ by `σ_scaled · target_scaler.scale_[0]`.
9. Save `data/processed/new_candidate_point_cat_{c}.csv` with the
   physical-scale ply values plus `Predicted_Strength_Unscaled,
   Predicted_Sigma_Unscaled, Predicted_Volume, Strategy_Score_Raw,
   Multi_Obj_Score, Selection_Mode`.

### 4.5 `visualize_patches.generate_candidate_visualizations`
- Loads each `new_candidate_point_cat_{c}.csv`, replays the ply orientations
  from `get_run_title` mapping, and emits:
  - A 3D staircase Plotly figure (HTML) of the stepped patch.
  - A 2D top-view matplotlib figure of the stacked ellipses.
- Hard-coded axis limit ±21 mm.

### 4.6 `update_data.update_data_file`
- Backs up `Data_v4.xlsx` to `Data_v4.xlsx.bak`.
- Reads existing sheets, captures their column order, appends the row from
  `new_candidate_point_cat_{c}.csv` with the **manually entered** new
  `Strength` value and the model's `Predicted_Strength_Mean`. Recomputes
  `Volume` using hard-coded thickness 2.0 mm.
- Writes both sheets back to the same file.

### 4.7 `main_orchestrator.main_orchestrator`
CLI: `python main_orchestrator.py --mode {active_learning|pure_mean|robust}`
(default `active_learning`).
Sequence: setup directories → STEP 1 prepare → STEP 2 train → STEP 3 find
(mode) → STEP 4 visualize → print the selected candidate row.

`PROJECT.CLEANUP_PREVIOUS_RUNS: False` keeps prior artifacts; flip to `True`
to wipe `data/processed/` and `artifacts/` first.

### 4.8 `run_final_inference.main`
CLI: `python run_final_inference.py --mode {...}` (default `robust`).
Skips prepare + train; loads existing artifacts; runs find + visualize. Use
this when you want to query the trained model with a different acquisition
mode without retraining.

## 5. Notebooks

### `paper_results.ipynb`
Generates the headline figures: physics-based selection (minimum volume with
`Strength ≥ ~70 kN`), strength/volume efficiency analysis, top-N candidate
tables per category, duplicate-design QC. Outputs
`final_result_cat{0,1}_physics.png`,
`pre_activelearning_result_cat{0,1}_physics.png`,
`parameter_evolution_cat{0,1}.png`, `performance_triplet_cat{0,1}.png`,
`top_view_Cat{0,1}_RowN.png`.

### `ablation_study.ipynb`
Five-variant ablation with a fixed validation split (Cat 1 by default):

| Name | depth | skip | kernel |
|------|-------|------|--------|
| Final DKGP (Deep + Matérn + Skip) | 5 | 2 | matern |
| DKGP (RBF Kernel) | 5 | 2 | rbf |
| Shallow DKGP (2 Layers) | 2 | 0 | matern |
| DKGP w/o Skip Connections | 5 | 0 | matern |
| Standard GP | — | — | matern (with `ard_num_dims`) |

Trains each for 6000 epochs across multiple seeds, tracks RMSE and NLL, and
emits `ablation_nll_curve.png` / `ablation_rmse_curve.png`. The depth-5 sweep
in the final cell is the configuration used for the paper's Table 4.

## 6. Two-stage training note

The published runs followed a two-stage strategy executed **manually**:
~38 initial seeds for Category 0 (28 for Category 1) were trained with a
shallow MLP + RBF kernel, then `config.yaml` was edited at N ≈ 70 to switch
to the deep MLP + Matérn 5/2 kernel and the pipeline was re-run. There is no
automatic phase switch in the code; the swap is a human-in-the-loop config
edit at the dataset-size boundary.

## 7. How to run

```bash
# Full run (prepare + train + find + visualize)
python main_orchestrator.py --mode active_learning

# Re-run only inference with a different acquisition strategy
python run_final_inference.py --mode robust    # LCB
python run_final_inference.py --mode pure_mean # mu only

# After a real FEM result is back, edit Strength in the candidate CSV, then:
python update_data.py        # appends the new row into Data_v4.xlsx (.bak created)

# Notebooks
jupyter lab paper_results.ipynb   # paper figures
jupyter lab ablation_study.ipynb  # Table 4 ablation
```

Dependencies are listed in `requirements.txt`.
