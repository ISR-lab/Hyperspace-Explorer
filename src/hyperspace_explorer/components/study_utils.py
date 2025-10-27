"""
Utility functions for study reproducibility, data loading, and LightGBM dataset creation.

This module provides helper functions for suppressing warnings, setting random seeds,
configuring plotting backends, preparing output directories, and loading datasets.
All functions are designed for reproducibility and clarity in scientific workflows.
"""

import warnings
from typing import Optional

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from IPython import get_ipython
from matplotlib import use
from matplotlib_inline.backend_inline import set_matplotlib_formats

from .config import Config


def suppress_noise() -> None:
    """
    Suppress Optuna and LightGBM warnings for cleaner output.
    """
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    warnings.filterwarnings(
        "ignore",
        message=".*LightGBM binary classifier with TreeExplainer shap values output has changed.*",
    )


def set_determinism(seed: int) -> np.random.Generator:
    """
    Set random seed for reproducibility across NumPy and LightGBM.

    Parameters
    ----------
    seed : int
        Random seed value.

    Returns
    -------
    np.random.Generator
        NumPy random number generator instance.
    """
    np.random.seed(seed)
    return np.random.default_rng(seed=seed)


def configure_matplotlib_backend(backend: str) -> None:
    """
    Configure the matplotlib backend for plotting.

    Parameters
    ----------
    backend : str
        Backend name (e.g., 'Agg', 'module://backend_interagg').
    """
    use(backend)
    if get_ipython():
        set_matplotlib_formats("retina")


def prepare_output_dirs(config: Config) -> None:
    """
    Prepare output directories for results and artifacts.

    Parameters
    ----------
    config : Config
        Configuration object with folder paths.
    """
    config.folder_name.mkdir(exist_ok=True)
    config.artifacts_folder.mkdir(exist_ok=True)
    for file_path in config.artifacts_folder.iterdir():
        if file_path.is_file():
            file_path.unlink()


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load a dataset from an Excel file.

    Parameters
    ----------
    file_path : str
        Path to the Excel file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """
    print(f"Loading dataset from {file_path}...")
    try:
        data = pd.read_excel(file_path)
        print(f"Data Loaded. Shape: {data.shape}")
        return data
    except FileNotFoundError:
        print("❌ Error: The input file was not found.")
        exit(1)


def create_dataset(
    data: pd.DataFrame, target: pd.Series, features: Optional[list[str]] = None
) -> lgb.Dataset:
    """
    Create a LightGBM Dataset for training or validation.

    Parameters
    ----------
    data : pd.DataFrame
        Feature matrix.
    target : pd.Series
        Target variable.
    features : list of str, optional
        List of features to include. If None, use all columns.

    Returns
    -------
    lgb.Dataset
        LightGBM Dataset object.
    """
    if features:
        data = data.loc[:, features]
    return lgb.Dataset(data, label=target, free_raw_data=False)
