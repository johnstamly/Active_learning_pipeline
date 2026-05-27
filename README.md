# Deep Kernel Bayesian Optimization for Stepped Composite Repairs

Companion code and data for the paper:

> **Deep Kernel Bayesian Optimization for the Design of Stepped Composite
> Repairs under Compression**
> Giannis Stamatelatos, Emilien Billaudeau, Maëlle Sergolle, Thomas Balutch,
> Vassilis Kostopoulos, Theodoros Loutas, Spyridon Psarras.
> *Composites Part A: Applied Science and Manufacturing* — **currently under
> peer review.**

> **Note**: The manuscript PDF is not distributed in this repository while the
> paper is under review. A link to the published version, together with the
> final DOI, will be added once the article is accepted.

---

## Overview

This repository implements a **Deep Kernel Bayesian Optimization** framework
for the geometric design of stepped composite repairs under compressive loading.
A Deep Kernel Gaussian Process (DKGP) surrogate is trained on a small set of
high-fidelity Cohesive Zone Modeling (CZM) Finite Element simulations and used
inside a Bayesian active-learning loop (Expected Improvement with a volume
penalty) to propose the next FEM run.

Two patch categories are handled in parallel:

- **Category 0** — aligned-orientation patch plies (single nesting constraint).
- **Category 1** — mixed off-axis patch plies (additional accessibility
  constraint imposed by top-down laser scarfing).

Validated against parametric CZM simulations, the framework reduces
computational cost by approximately **96 %** relative to typical Genetic
Algorithm budgets reported for problems of comparable dimensionality, and by
**80 %** relative to static (Latin-Hypercube) sampling. The optimal designs
recover ≈ 95 % of the parent laminate's compressive strength.

## Repository contents

```
config.yaml                   # paths and hyperparameters
config_manager.py             # dot-accessor for config.yaml
prepare_data.py               # Excel -> processed_data.csv (+ target scaler)
model_definitions.py          # DeepKernel (MLP) + DeepKernelGP (GPyTorch)
train_surrogate.py            # one DKGP model per category
find_new_candidate_points.py  # Bayesian acquisition (EI / pure_mean / robust)
visualize_patches.py          # 3D staircase + 2D top-view
main_orchestrator.py          # full pipeline CLI
run_final_inference.py        # inference-only CLI
update_data.py                # append a real FEM result back into the dataset
cleanup_volume.py             # recompute Volume column for existing rows
paper_results.ipynb           # paper figures (Pareto/L-shape, top-N tables)
ablation_study.ipynb          # Table 4 ablation (RMSE, NLL)
data/Data_v4.xlsx             # raw FEM dataset (Elips0 + Elips1 sheets)
data/processed/               # processed_data.csv, candidate CSVs
artifacts/                    # saved DKGP weights, train tensors, target scaler
REPO_NOTES.md                 # contributor-facing deep dive
```

## Installation

Python 3.8+ is recommended. Install dependencies:

```bash
git clone https://github.com/<your-org>/active_learning_pipeline.git
cd active_learning_pipeline
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The pinned versions in `requirements.txt` match the environment used to
produce the published results.

## Quickstart

```bash
# Full pipeline: prepare data, train surrogates, run active learning, visualize
python main_orchestrator.py --mode active_learning
```

The orchestrator runs:

1. **Prepare** — read `data/Data_v4.xlsx`, scale features, write
   `data/processed/processed_data.csv` and `artifacts/target_scaler.pkl`.
2. **Train** — fit one DKGP per category (10 000 epochs by default), save
   weights and training tensors under `artifacts/`.
3. **Find** — Monte Carlo sample 2 × 10⁶ feasible candidate geometries,
   score them with the chosen acquisition function, write
   `data/processed/new_candidate_point_cat_{0,1}.csv`.
4. **Visualize** — emit 3D staircase Plotly and 2D top-view matplotlib
   figures for each winning candidate.

To run inference only (reusing saved artifacts) with a different acquisition
mode:

```bash
python run_final_inference.py --mode robust       # LCB (μ − 1.96σ)
python run_final_inference.py --mode pure_mean    # μ only
python run_final_inference.py --mode active_learning   # Expected Improvement
```

After a real FEM simulation comes back, edit the `Strength` value in the
appropriate `new_candidate_point_cat_{c}.csv` and run:

```bash
python update_data.py    # appends the new row into data/Data_v4.xlsx (backup created)
```

## Reproducing the paper figures

```bash
jupyter lab paper_results.ipynb   # Pareto/L-shape figures, top-N tables
jupyter lab ablation_study.ipynb  # Table 4 ablation curves
```

`paper_results.ipynb` writes its outputs as PNGs into the repository root
(`final_result_cat{0,1}_physics.png`, `parameter_evolution_cat{0,1}.png`,
`performance_triplet_cat{0,1}.png`, `top_view_Cat{0,1}_RowN.png`, etc.).

## Data

`data/Data_v4.xlsx` contains the raw FEM dataset, with two sheets:

- **Elips0** — Category 0 patches (aligned orientation).
- **Elips1** — Category 1 patches (mixed 0°/90°).

The 12 geometric design variables (`a_ply1..6, b_ply1..6`) are integer
semi-axis lengths in [5, 35] mm. `Strength` is the ultimate compressive
strength in kN; `Volume` is the excavated repair volume in mm³ computed with
a 2.0 mm ply thickness. See `REPO_NOTES.md` §3 for the full column schema.

The high-fidelity FEM input decks and detailed output databases are available
from the corresponding author on reasonable request, as stated in the
manuscript's Data Availability section.

## Citation

The paper is currently under peer review at *Composites Part A: Applied
Science and Manufacturing*. Citation details, including the DOI, will be
added here once the article is accepted. In the meantime, if you wish to
reference this work, please contact the corresponding author.

A `CITATION.cff` file is provided as a placeholder and will be updated upon
publication.

## License

This project is released under the MIT License — see [`LICENSE`](LICENSE).

## Contact

Corresponding author: Giannis Stamatelatos — `mead6828@ac.upatras.gr`
Applied Mechanics Laboratory, Department of Mechanical Engineering &
Aeronautics, University of Patras, Greece.
