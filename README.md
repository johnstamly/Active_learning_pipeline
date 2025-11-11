# Active Learning Pipeline for Composite Elliptical Patches

This project implements an active learning pipeline to optimize material strength in composite elliptical patches using surrogate modeling and Bayesian optimization. It targets two categories of patches (Elips0 and Elips1) with design variables including ply thicknesses (a_ply1-6 and b_ply1-6). The pipeline predicts and maximizes the "Strength" property while respecting monotonic and cross-ply constraints.

## Workflow

1. **Data Preparation**: Loads data from `Data_v4.xlsx`, merges sheets, adds a category feature, cleans the data, scales the target variable, and saves the processed data and scaler.

2. **Surrogate Model Training**: Trains two DKGP (Deep Kernel Gaussian Process) models per category using the processed data, then saves the trained models and tensors.

3. **Candidate Finding**: Performs Bayesian optimization to identify new candidate points that maximize the Expected Improvement (EI), saving the candidates as CSV files.

## Main Components

- **ConfigManager**: Handles configuration settings from `config.yaml`.
- **Data Handling**: Utilizes Pandas and scikit-learn for data processing and scaling.
- **Modeling**: Employs PyTorch and GPyTorch for building and training surrogate models.
- **Optimization**: Uses SciPy and NumPy for Bayesian optimization routines.
- **Orchestrator Script**: `main_orchestrator.py` coordinates the entire pipeline execution.

## Usage Instructions

1. Install the required dependencies:
   - PyTorch
   - GPyTorch
   - Pandas
   - NumPy
   - SciPy
   - scikit-learn
   - joblib

2. Ensure `config.yaml` is properly configured with the necessary settings.

3. Run the pipeline:
   ```
   python main_orchestrator.py
   ```

4. Outputs will be generated in the `data/processed/` and `artifacts/` directories, including processed data, trained models, and new candidate points.