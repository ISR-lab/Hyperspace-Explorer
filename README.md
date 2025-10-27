# Hyperspace Explorer

Hyperspace Explorer is a toolkit for running, optimizing, and visualizing machine learning experiments—especially with LightGBM and Optuna. It streamlines data preprocessing, hyperparameter tuning, and result analysis in a reproducible workflow.

## Requirements
- Python 3.9+
- pip
- pip-compile (from `pip-tools`)
- Docker (recommended)
- (Optional) Jupyter Lab (for local, non-Docker use)

## Tools Used
- **LightGBM**: Fast, efficient gradient boosting framework for machine learning tasks, especially suited for large datasets and tabular data.
- **Optuna**: Hyperparameter optimization framework that automates the search for optimal model parameters.
- **Jupyter Lab**: Interactive development environment for running and visualizing experiments in notebooks.
- **pip-tools (pip-compile)**: For managing and compiling precise Python dependency requirements.
- **Docker**: Ensures reproducible environments and easy deployment across systems.

## Running with Docker (Recommended)
1. Build the Docker image:
   ```bash
   make build
   ```
2. Start the container:
   ```bash
   make run
   ```
3. Open your browser at [http://localhost:8888/lab](http://localhost:8888/lab) to access Jupyter Lab.

## Running Locally (Without Docker)
1. Install pip-compile if not already installed:
   ```bash
   pip install pip-tools
   ```
2. Compile requirements:
   ```bash
   pip-compile requirements.in
   pip-compile requirements-dev.in
   ```
3. Install requirements:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt 
   ```
4. Start Jupyter Lab:
   ```bash
   jupyter lab
   ```
5. Open your browser at [http://localhost:8888/lab](http://localhost:8888/lab).

## Project Structure
- `notebooks/`: Jupyter notebooks for each experiment (Pain, Disability, QoL, PGIC).
- `configs/`: JSON configuration files for each experiment.
- `data/input/`: Placeholder for raw data (e.g., `raw_new.xlsx`). **Not versioned.**
- `data/outputs/`: Directory for all generated outputs (figures, models, results). **Not versioned.**
- `src/`: Source code for data preprocessing and utilities.
- `Dockerfile`: Defines the reproducible environment for running the analysis.
- `LICENSE`: MIT License.

## Citation
If you use this code or find it helpful in your research, please cite our paper:

> Meier, M., Wirth, B., Wehrli, M., et al. (2025). *A transparent and reproducible machine learning framework for small datasets*. [Journal Name], [Volume](Issue), pages.

## Notes
- Notebooks are in the `notebooks/` directory.
- Configurations are in `configs/`. Each JSON config file defines the experiment setup, including:
  - Path to the input data file (must be XLSX format, e.g., `raw_new.xlsx`)
  - Output directory for results
  - Features (categorical/numerical), target variable, and experiment parameters (cross-validation, trials, etc.)
  - Plotting and reproducibility settings
- Data should be placed in `data/input/` and must be an Excel `.xlsx` file matching the config.
- Outputs (figures, models, results) are written to the `outputs/` directory and will be visible on your host machine when running in Docker.
- Each experiment's output folder contains a `models/` directory, where trained model artifacts are stored for later analysis or reuse.