# Active Learning Pipeline for Composite Elliptical Patches

## Project Overview
This repository implements an **active learning pipeline** to optimize material strength in composite elliptical patches using surrogate modeling and Bayesian optimization.  
It focuses on two patch categories (Elips0 and Elips1) with design variables comprising ply thicknesses (`a_ply1-6` and `b_ply1-6`). The pipeline predicts and maximizes the **Strength** property while respecting monotonic and cross‑ply constraints.

## Installation
1. **Clone the repository**  
   ```bash
   git clone https://github.com/yourusername/active_learning_pipeline.git
   cd active_learning_pipeline
   ```
2. **Set up a Python environment** (recommended with `conda` or `venv`)  
   ```bash
   conda create -n alp python=3.10
   conda activate alp
   # or: python -m venv venv && source venv/bin/activate
   ```
3. **Install required packages**  
   - The project depends on: `torch`, `gpytorch`, `pandas`, `numpy`, `scipy`, `scikit-learn`, `joblib`.  
   - Install them via pip (or add a `requirements.txt` file and run `pip install -r requirements.txt`):  
   ```bash
   pip install torch gpytorch pandas numpy scipy scikit-learn joblib
   ```
4. **Data preparation**  
   - Run the data‑preparation script to generate processed data and scalers:  
   ```bash
   python prepare_data.py
   ```
   - This creates `data/processed/processed_data.csv` and stores scaler objects in `artifacts/`.

## Usage
### Notebooks
- **Paper results**: `paper_results.ipynb` – reproduces the main figures and tables presented in the manuscript.  
- **Ablation study**: `ablation_study.ipynb` – runs ablation experiments and visualises their impact.

Open the notebooks with Jupyter (e.g., `jupyter lab`) and execute all cells.

### Python scripts
| Script | Description |
|--------|-------------|
| `train_surrogate.py` | Trains the DKGP surrogate models for each patch category. |
| `run_final_inference.py` | Performs Bayesian optimization to find new candidate points and saves them as CSV files. |
| `prepare_data.py` | Pre‑processes the raw Excel data (`Data_v4.xlsx`) into a clean CSV. |
| `find_new_candidate_points.py` | Utility to explore candidate points after training. |
| `visualize_patches.py` | Generates visualisations of patch designs. |
| `cleanup_volume.py` | Helper script to prune intermediate artefacts. |

Typical workflow:
```bash
python prepare_data.py           # data preparation
python train_surrogate.py       # model training
python run_final_inference.py   # candidate selection
```

## Data Description
- **`data/processed/processed_data.csv`** – clean dataset used for training.  
- **`data/figures/`** – contains visualisations:
  - `figures/top_view_2d/` – 2‑D top‑view images per category and row.  
  - `figures/staircase_3d/` – interactive HTML stair‑case visualisations.  
- Additional CSV files (`new_candidate_point_cat_0.csv`, `new_candidate_point_cat_1.csv`) store the optimisation results.

## Results
Key outcome figures are stored in the repository root:

- `final_result_cat0.png` / `final_result_cat1.png` – final strength predictions per category.  
- `final_result_cat0_physics.png` / `final_result_cat1_physics.png` – physics‑based visualisation of the optimal designs.  
- `performance_triplet_cat0.png` / `performance_triplet_cat1.png` – performance over training epochs.  
- `ablation_nll_curve.png` & `ablation_rmse_curve.png` – ablation study loss curves.  
- `parameter_evolution_cat0.png` / `parameter_evolution_cat1.png` – evolution of design parameters.  

Refer to the notebooks for detailed generation steps.

## Citation
If you use this repository in your research, please cite it as follows:

```
Author(s). (Year). *Active Learning Pipeline for Composite Elliptical Patches*. GitHub repository. https://github.com/yourusername/active_learning_pipeline. DOI: 10.1234/yourdoi
```

Replace the placeholder DOI with the actual one when available.

## License
This project is licensed under the **MIT License**. See the `LICENSE` file for full details.

---

### Additional Notes
- **Workflow Summary** – Data preparation → Surrogate model training → Bayesian optimization → Candidate extraction.  
- **Main Components** – `ConfigManager` (handles `config.yaml`), data handling (Pandas, scikit‑learn), modeling (PyTorch & GPyTorch), optimization (SciPy, NumPy), orchestration (`main_orchestrator.py`).  

For any questions or contributions, please open an issue or submit a pull request.