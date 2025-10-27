"""
Configuration class for LightGBM/Optuna study experiments.

This module provides a Config class for loading and managing all experiment settings
from a JSON configuration file. It ensures reproducibility and clarity for scientific workflows.
"""

import json
from pathlib import Path
from typing import Any, Dict


class Config:
    """
    Configuration class for managing experiment settings.

    Loads all parameters, file paths, and feature lists from a JSON config file.
    """

    scale_pos_weight: Any = None

    def __init__(self, config_path: Path):
        """
        Initialize Config from a JSON file.

        Parameters
        ----------
        config_path : Path
            Path to the JSON configuration file.
        """
        with config_path.open("r", encoding="utf-8") as file:
            study_def = json.load(file)

        # Files and folders
        self.file_name: str = study_def["source_file"]
        self.folder_name: Path = Path(study_def["output_dir"])
        self.artifacts_folder: Path = Path(study_def["output_dir"], "models")

        # Define features
        self.features: list[str] = study_def["features"]
        self.categorical_features: list[str] = study_def["categorical_features"]
        self.numerical_features: list[str] = study_def["numerical_features"]
        self.target_source: str = study_def["target_source"]
        self.target: str = study_def["target"]

        # Basic config
        self.test_split: float = study_def["test_split"]
        self.n_trials: int = study_def["n_trials"]
        self.cv_splits: int = study_def["cv_splits"]
        self.early_stopping: int = study_def["early_stopping"]
        self.n_final_trials: int = study_def["n_final_trials"]
        self.dpi: int = study_def["dpi"]
        self.feat_count_cov: float = study_def["feature_count_coverage"]
        self.bin_size: int = study_def["bin_size"]
        self.hold_out: float = study_def["hold_out"]
        self.random_seed: int = study_def["random_seed"]
        self.plt_backend: str = study_def["plt_backend"]
        self.num_boost_round: int = study_def.get("num_boost_round", 500)

    def get_base_params(self) -> Dict[str, Any]:
        """
        Returns a dictionary of base LightGBM parameters for binary classification.
        """
        return {
            "verbose": -1,
            "objective": "binary",
            "n_jobs": 1,
            "num_threads": 1,
            "tree_learner": "serial",
            "deterministic": True,
            "seed": self.random_seed,
            "feature_fraction_seed": self.random_seed,
            "data_random_seed": self.random_seed,
            "max_bin": 255,
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "is_unbalance": False,
            "scale_pos_weight": self.scale_pos_weight,
            "force_col_wise": True,
        }

    def set_loocv_params(
        self, trial: Any, numerical_param_df: Any, categorical_param_df: Any
    ) -> Dict[str, Any]:
        """
        Return a complete parameter dict by merging base params with Optuna-suggested values.
        """
        param = self.get_base_params()
        param.update(
            {
                "subsample": trial.suggest_float(
                    "subsample",
                    numerical_param_df.loc["min", "subsample"],
                    numerical_param_df.loc["max", "subsample"],
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    numerical_param_df.loc["min", "learning_rate"],
                    numerical_param_df.loc["max", "learning_rate"],
                ),
                "feature_fraction": trial.suggest_float(
                    "feature_fraction",
                    numerical_param_df.loc["min", "feature_fraction"],
                    numerical_param_df.loc["max", "feature_fraction"],
                ),
                "max_depth": trial.suggest_int(
                    "max_depth",
                    int(numerical_param_df.loc["min", "max_depth"]),
                    int(numerical_param_df.loc["max", "max_depth"]),
                    step=2,
                ),
                "min_child_weight": trial.suggest_int(
                    "min_child_weight",
                    int(numerical_param_df.loc["min", "min_child_weight"]),
                    int(numerical_param_df.loc["max", "min_child_weight"]),
                ),
                "grow_policy": trial.suggest_categorical(
                    "grow_policy",
                    categorical_param_df["grow_policy"].dropna().unique().tolist(),
                ),
            }
        )
        return param

    def set_ensemble_params(
        self, trial: Any, numerical_param_df: Any, categorical_param_df: Any
    ) -> Dict[str, Any]:
        """
        Return a complete parameter dict for ensemble training by merging base params with Optuna-suggested values.
        """
        param = self.get_base_params()
        param.update(
            {
                "subsample": trial.suggest_float(
                    "subsample",
                    numerical_param_df.loc["min", "subsample"],
                    numerical_param_df.loc["max", "subsample"],
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    numerical_param_df.loc["min", "learning_rate"],
                    numerical_param_df.loc["max", "learning_rate"],
                ),
                "feature_fraction": trial.suggest_float(
                    "feature_fraction",
                    numerical_param_df.loc["min", "feature_fraction"],
                    numerical_param_df.loc["max", "feature_fraction"],
                ),
                "max_depth": trial.suggest_int(
                    "max_depth",
                    numerical_param_df.loc["min", "max_depth"],
                    numerical_param_df.loc["max", "max_depth"],
                    step=2,
                ),
                "min_child_weight": trial.suggest_int(
                    "min_child_weight",
                    numerical_param_df.loc["min", "min_child_weight"],
                    numerical_param_df.loc["max", "min_child_weight"],
                ),
                "grow_policy": trial.suggest_categorical(
                    "grow_policy",
                    categorical_param_df["grow_policy"].dropna().unique().tolist(),
                ),
            }
        )
        return param
