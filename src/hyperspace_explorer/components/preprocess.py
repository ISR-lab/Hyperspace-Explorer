"""
Preprocessing utilities for data cleaning, feature validation, and splitting.

This module provides functions for validating features, handling categorical variables,
creating target variables, reporting missing values, splitting data for holdout evaluation,
and calculating class imbalance. All functions are designed for reproducibility and clarity
in scientific workflows.
"""

from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def validate_features(data: pd.DataFrame, expected_features: List[str]) -> List[str]:
    """
    Validate that all expected features are present in the DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    expected_features : list of str
        List of expected feature names.

    Returns
    -------
    list of str
        List of existing features found in the data.
    """
    existing = data.columns.intersection(expected_features)
    missing = set(expected_features) - set(existing)
    if missing:
        print(f"⚠️ Missing features: {missing}")
    return list(existing)


def set_categorical(
    data: pd.DataFrame, categorical_features: List[str]
) -> pd.DataFrame:
    """
    Set specified columns as categorical and impute missing values.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    categorical_features : list of str
        List of categorical feature names.

    Returns
    -------
    pd.DataFrame
        DataFrame with categorical columns set.
    """
    existing = data.columns.intersection(categorical_features)
    data[existing] = data[existing].astype("category")
    return data


def create_target(data: pd.DataFrame, target_source: str, target: str) -> pd.DataFrame:
    """
    Create a binary target variable from a source column.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    target_source : str
        Source column for target creation.
    target : str
        Name of the new target column.

    Returns
    -------
    pd.DataFrame
        DataFrame with the new target column.
    """
    data = data[~data[target_source].isna()].copy()
    data[target] = data[target_source].isin([1, 2]).astype(int)
    if target not in data.columns:
        raise ValueError(f"❌ Target variable '{target}' is missing.")
    return data


def report_missing_values(data: pd.DataFrame, features: List[str]) -> None:
    """
    Report missing values for specified features.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    features : list of str
        List of feature names to check.
    """
    missing_counts = data[features].isnull().sum()
    for col, count in missing_counts[missing_counts > 0].items():
        print(f"⚠️ Column '{col}' has {count} missing values.")


def split_data(
    X: pd.DataFrame, y: pd.Series, holdout_ratio: float, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training/test and holdout sets with stratification.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable.
    holdout_ratio : float
        Proportion of data to use as holdout set.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        (X_train, X_holdout, y_train, y_holdout)
    """
    X, holdout_X, y, holdout_y = train_test_split(
        X, y, test_size=holdout_ratio, random_state=seed, stratify=y
    )

    print(f"\nNumber of records in the training/test set: {len(X)}")
    print(f"Number of records in the holdout set: {len(holdout_X)}")

    print("\nClass distribution in the training/test set:")
    print(y.value_counts())

    print("\nClass distribution in the holdout set:")
    print(holdout_y.value_counts())
    return X, holdout_X, y, holdout_y
