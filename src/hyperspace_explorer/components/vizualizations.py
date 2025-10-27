"""
Visualization utilities for model evaluation and feature analysis.

This module provides functions to generate publication-quality plots for feature selection,
model performance, and SHAP-based interpretability in machine learning studies.
All plots are saved in SVG format for high-quality inclusion in scientific papers.
"""

from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from IPython.display import SVG, display
from matplotlib.pyplot import Figure
from sklearn.metrics import roc_curve, auc


def plot_feature_amount_frequency_loocv(
    feature_amount_freq: pd.Series,
    cumulative_percentages: pd.Series,
    x_positions: np.ndarray,
    config: Any,
    threshold_pct: float,
    fig_name: str = "s2_feature_amount_frequency.svg",
) -> Figure:
    """
    Plot a Pareto histogram showing the number of features selected per trial.

    Parameters
    ----------
    feature_amount_freq : pd.Series
        Frequency of feature counts per trial.
    cumulative_percentages : pd.Series
        Cumulative percentage of feature counts.
    x_positions : np.ndarray
        X-axis positions for bars.
    config : Any
        Configuration object with output folder and DPI.
    threshold_pct : float
        Threshold percentage for cumulative selection.
    fig_name : str, optional
        Output filename for the figure.

    Returns
    -------
    Figure
        The matplotlib Figure object.
    """
    figfile_path = config.folder_name / fig_name
    threshold_idx = cumulative_percentages.searchsorted(threshold_pct * 100)

    fig, ax1 = plt.subplots(figsize=(10, (9 / 16) * 10))

    # Bar segments: below/above threshold
    bars1 = ax1.bar(
        x_positions[: threshold_idx + 1],
        feature_amount_freq.iloc[: threshold_idx + 1],
        color="orange",
        alpha=0.7,
        label="Selected FeatureCounts",
    )
    bars2 = ax1.bar(
        x_positions[threshold_idx + 1 :],
        feature_amount_freq.iloc[threshold_idx + 1 :],
        color="C0",
        alpha=0.7,
    )

    # Primary axis styling
    ax1.set_xlabel("Number of Features Selected")
    ax1.set_ylabel("Frequency", color="C0")
    ax1.set_xticks(x_positions)
    ax1.tick_params(axis="y", labelcolor="C0")

    for bar in bars1 + bars2:
        yval = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.05,
            str(int(yval)),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Secondary axis: cumulative %
    ax2 = ax1.twinx()
    ax2.plot(x_positions, cumulative_percentages, color="C1", marker="o")
    ax2.set_ylabel("Cumulative Percentage (%)", color="C1")
    ax2.set_ylim(0, 110)
    ax2.axhline(y=threshold_pct * 100, color="red", linestyle="--")
    ax2.text(
        x=0.98,
        y=threshold_pct * 100 + 1,
        s=f"{threshold_pct * 100:.0f}%",
        ha="right",
        va="bottom",
        transform=ax2.get_yaxis_transform(),
        color="red",
        fontsize=10,
    )

    plt.title("Pareto Histogram (Number of Features Selected Per Trial)")
    fig.legend(loc="upper left", bbox_to_anchor=(0.7, 0.9))
    plt.tight_layout()
    plt.savefig(figfile_path, dpi=config.dpi, format="svg")
    display(SVG(filename=figfile_path))
    plt.close()

    return fig


def plot_feature_selection_frequency_loocv(
    frequency_df_sorted: pd.DataFrame,
    selected_features_coverage_pct: float,
    config: Any,
    fig_name: str = "s2_feature_selection_frequency.svg",
) -> Figure:
    """
    Plot a Pareto chart of feature selection frequency across trials.

    Parameters
    ----------
    frequency_df_sorted : pd.DataFrame
        DataFrame with feature frequencies and selection flags.
    selected_features_coverage_pct : float
        Percentage of selections covered by final features.
    config : Any
        Configuration object with output folder and DPI.
    fig_name : str, optional
        Output filename for the figure.

    Returns
    -------
    Figure
        The matplotlib Figure object.
    """
    figfile_path = config.folder_name / fig_name
    fig, ax1 = plt.subplots(figsize=(10, (9 / 16) * 10))

    bar_colors = frequency_df_sorted["Selected"].map({True: "orange", False: "skyblue"})
    bars = ax1.bar(
        frequency_df_sorted["Feature"],
        frequency_df_sorted["Frequency"],
        color=bar_colors,
        alpha=0.7,
    )

    ax1.set_ylabel("Selection Frequency")
    ax1.set_xlabel("Features")
    ax1.set_title("Feature Selection Frequency Across Trials (Pareto-like Chart)")

    for bar in bars:
        yval = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.05,
            str(int(yval)),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Secondary axis: cumulative %
    ax2 = ax1.twinx()
    ax2.plot(
        frequency_df_sorted["Feature"],
        frequency_df_sorted["Cumulative_Percentage"],
        color="C1",
        marker="o",
    )
    ax2.set_ylabel("Cumulative Percentage (%)", color="C1")
    ax2.axhline(y=selected_features_coverage_pct, color="red", linestyle="--")
    ax2.text(
        x=0.98,
        y=selected_features_coverage_pct + 1,
        s=f"{selected_features_coverage_pct:.0f}%",
        ha="right",
        va="bottom",
        transform=ax2.get_yaxis_transform(),
        color="red",
        fontsize=10,
    )

    plt.setp(ax1.get_xticklabels(), rotation=90, ha="center")
    fig.legend(["Selected Features"], loc="upper left", bbox_to_anchor=(0.7, 0.9))
    plt.tight_layout()
    plt.savefig(figfile_path, dpi=config.dpi, format="svg")
    display(SVG(filename=figfile_path))
    plt.close()

    return fig


def plot_average_roc_curve_loocv(
    final_study: Any, y_true: np.ndarray, config: Any
) -> None:
    """
    Plot the average ROC curve using LOOCV with 95% CI bootstrap over individual folds.

    Parameters
    ----------
    final_study : Any
        Optuna study object containing best trials.
    y_true : np.ndarray
        True binary labels.
    config : Any
        Configuration object with output folder and DPI.
    """
    mean_fpr = np.linspace(0, 1, 100)
    
    # Get the best trial (typically only one in LOOCV)
    best_trial = final_study.best_trials[0] if final_study.best_trials else None
    if best_trial is None:
        print("No best trial found for ROC curve.")
        return
    
    y_score = best_trial.user_attrs.get("predictions")
    if y_score is None:
        print("No predictions found for ROC curve.")
        return
    
    # Convert to numpy arrays to avoid pandas indexing issues
    y_true_array = np.array(y_true)
    y_score_array = np.array(y_score)
    
    # Calculate overall ROC curve
    fpr, tpr, _ = roc_curve(y_true_array, y_score_array)
    mean_auc = auc(fpr, tpr)
    
    # Bootstrap 95% confidence intervals over LOOCV folds (individual predictions)
    n_bootstraps = 1000
    rng = np.random.RandomState(config.random_seed if hasattr(config, 'random_seed') else 42)
    bootstrapped_aucs = []
    bootstrapped_tprs = []
    
    n_samples = len(y_true_array)
    
    for i in range(n_bootstraps):
        # Resample indices with replacement (bootstrap over individual LOOCV folds)
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        
        # Get bootstrapped samples
        y_true_boot = y_true_array[indices]
        y_score_boot = y_score_array[indices]
        
        # Skip if only one class present in bootstrap sample
        if len(np.unique(y_true_boot)) < 2:
            continue
        
        # Calculate ROC for this bootstrap sample
        try:
            fpr_boot, tpr_boot, _ = roc_curve(y_true_boot, y_score_boot)
            interp_tpr = np.interp(mean_fpr, fpr_boot, tpr_boot)
            interp_tpr[0] = 0.0
            interp_tpr[-1] = 1.0
            bootstrapped_tprs.append(interp_tpr)
            bootstrapped_aucs.append(auc(fpr_boot, tpr_boot))
        except:
            continue
    
    # Calculate mean TPR for plotting
    interp_tpr_mean = np.interp(mean_fpr, fpr, tpr)
    interp_tpr_mean[0] = 0.0
    interp_tpr_mean[-1] = 1.0
    
    # Calculate 95% confidence intervals
    if len(bootstrapped_tprs) > 0:
        tprs_lower = np.percentile(bootstrapped_tprs, 2.5, axis=0)
        tprs_upper = np.percentile(bootstrapped_tprs, 97.5, axis=0)
        auc_lower = np.percentile(bootstrapped_aucs, 2.5)
        auc_upper = np.percentile(bootstrapped_aucs, 97.5)
    else:
        # Fallback if bootstrap failed
        tprs_lower = interp_tpr_mean
        tprs_upper = interp_tpr_mean
        auc_lower = mean_auc
        auc_upper = mean_auc

    figfile = config.folder_name / "s3_average_roc_curve.svg"
    plt.figure(figsize=(10, (9 / 16) * 10))
    plt.plot(
        mean_fpr,
        interp_tpr_mean,
        label=f"Mean ROC (AUC = {mean_auc:.3f}, 95% CI: [{auc_lower:.3f}-{auc_upper:.3f}])",
        color="b",
        linewidth=2,
    )
    plt.fill_between(
        mean_fpr, tprs_lower, tprs_upper, alpha=0.2, color="gray", label="95% CI (Bootstrap)"
    )
    plt.plot([0, 1], [0, 1], linestyle="--", color="red", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve with 95% CI (Bootstrap over LOOCV Folds)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(figfile, dpi=config.dpi, format="svg")
    display(SVG(filename=str(figfile)))
    plt.close()


def plot_shap_summary_loocv(
    best_trial: Any, final_features: List[str], X: pd.DataFrame, rng: Any, config: Any
) -> None:
    """
    Plot a SHAP summary plot for the best trial using LOOCV.

    Parameters
    ----------
    best_trial : Any
        Optuna trial object.
    final_features : List[str]
        List of selected features.
    X : pd.DataFrame
        Feature matrix.
    rng : Any
        Random number generator.
    config : Any
        Configuration object with output folder and DPI.
    """
    shap_values = np.stack(best_trial.user_attrs["all_shap_values"]).squeeze(axis=1)
    data = X[final_features]

    figfile = config.folder_name / "s3_shap_summary.svg"
    plt.figure(figsize=(10, (9 / 16) * 10))
    shap.summary_plot(
        shap_values, data, feature_names=final_features, show=False, rng=rng
    )
    plt.title("SHAP Summary Plot")
    plt.tight_layout()
    plt.savefig(figfile, dpi=config.dpi, format="svg")
    display(SVG(filename=str(figfile)))
    plt.close()


def plot_iterations_histogram_loocv(best_trial: Any, config: Any) -> None:
    """
    Plot a histogram of estimator iterations per LOOCV fold with cumulative curve.

    Parameters
    ----------
    best_trial : Any
        Optuna trial object.
    config : Any
        Configuration object with output folder and DPI.
    """
    iterations = best_trial.user_attrs.get("iterations", [])
    bins = np.arange(
        min(iterations), max(iterations) + config.bin_size, config.bin_size
    )
    counts, bin_edges = np.histogram(iterations, bins=bins)
    x_positions = (bin_edges[:-1] + bin_edges[1:]) / 2
    cumulative = np.cumsum(counts) / np.sum(counts) * 100

    figfile = config.folder_name / "s3_iterations_amount_per_fold_with_cumulative.svg"
    fig, ax1 = plt.subplots(figsize=(10, (9 / 16) * 10))
    bars = ax1.bar(x_positions, counts, width=config.bin_size, alpha=0.7, color="C0")
    ax1.set_xlabel(f"Iterations per LOOCV Fold (binsize={config.bin_size})")
    ax1.set_ylabel("Count", color="C0")
    ax1.set_xticks(x_positions)

    for bar in bars:
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{int(bar.get_height())}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax2 = ax1.twinx()
    ax2.plot(x_positions, cumulative, color="C1", marker="o")
    ax2.set_ylabel("Cumulative Percentage (%)", color="C1")
    ax2.set_ylim(0, 110)
    plt.title("Estimator Histogram with Cumulative Curve")
    fig.tight_layout()
    plt.savefig(figfile, dpi=config.dpi, format="svg")
    display(SVG(filename=str(figfile)))
    plt.close()


def plot_average_roc_curve_ensemble(
    best_trial: Any, config: Any, save_as: str = "s4_average_roc_curve.svg"
) -> None:
    """
    Plot the average ROC curve for an ensemble of models.

    Parameters
    ----------
    best_trial : Any
        Optuna trial object.
    config : Any
        Configuration object with output folder and DPI.
    save_as : str, optional
        Output filename for the figure.
    """
    roc_curves = best_trial.user_attrs["roc_curves"]
    figfile_name = config.folder_name / save_as

    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []

    for fpr, tpr, roc_auc in roc_curves:
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(roc_auc)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    plt.figure(figsize=(10, 6))
    plt.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=f"Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})",
        lw=2,
        alpha=0.8,
    )
    plt.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"± 1 std. dev.",
    )
    plt.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Random", alpha=0.8)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Average ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig(figfile_name, dpi=300, format="svg")
    display(SVG(filename=str(figfile_name)))
    plt.close()


def plot_shap_summary_ensemble(
    best_trial: Any,
    data: pd.DataFrame,
    final_features: List[str],
    config: Any,
    rng: Any,
    save_as: str = "s4_shap_summary.svg",
) -> None:
    """
    Plot a SHAP summary plot for an ensemble of models.

    Parameters
    ----------
    best_trial : Any
        Optuna trial object.
    data : pd.DataFrame
        Feature matrix.
    final_features : List[str]
        List of selected features.
    config : Any
        Configuration object with output folder and DPI.
    rng : Any
        Random number generator.
    save_as : str, optional
        Output filename for the figure.
    """
    shap_values = np.concatenate(best_trial.user_attrs["all_shap_values"], axis=0)
    figfile_name = config.folder_name / save_as

    plt.figure(figsize=(10, (9 / 16) * 10))
    shap.summary_plot(
        shap_values, data, feature_names=final_features, show=False, rng=rng
    )
    plt.title("SHAP Summary Plot")
    plt.tight_layout()
    plt.savefig(figfile_name, dpi=config.dpi, format="svg")
    display(SVG(filename=str(figfile_name)))
    plt.close()


def plot_shap_decision_ensemble(
    best_trial: Any,
    data: pd.DataFrame,
    final_features: List[str],
    config: Any,
    save_as: str = "s4_shap_decision_plot.svg",
) -> None:
    """
    Plot a SHAP decision plot for an ensemble of models.

    Parameters
    ----------
    best_trial : Any
        Optuna trial object.
    data : pd.DataFrame
        Feature matrix.
    final_features : List[str]
        List of selected features.
    config : Any
        Configuration object with output folder and DPI.
    save_as : str, optional
        Output filename for the figure.
    """
    shap_values = np.concatenate(best_trial.user_attrs["all_shap_values"], axis=0)
    figfile_name = config.folder_name / save_as

    plt.figure(figsize=(10, (9 / 16) * 10))
    shap.decision_plot(
        base_value=np.asarray(shap_values).mean(),
        shap_values=shap_values,
        features=data,
        feature_names=final_features,
        show=False,
    )
    plt.title("SHAP Decision Plot")
    plt.tight_layout()
    plt.savefig(figfile_name, dpi=config.dpi, format="svg")
    display(SVG(filename=str(figfile_name)))
    plt.close()


def plot_shap_dependence_ensemble(
    best_trial: Any,
    data: pd.DataFrame,
    final_features: List[str],
    config: Any,
    max_features: int = 3,
) -> None:
    """
    Plot SHAP dependence plots for the top features in an ensemble.

    Parameters
    ----------
    best_trial : Any
        Optuna trial object.
    data : pd.DataFrame
        Feature matrix.
    final_features : List[str]
        List of selected features.
    config : Any
        Configuration object with output folder and DPI.
    max_features : int, optional
        Number of top features to plot.
    """
    shap_values = np.concatenate(best_trial.user_attrs["all_shap_values"], axis=0)

    feature_importance_df = pd.DataFrame(
        {
            "feature": final_features,
            "importance": np.abs(shap_values).mean(axis=0),
        }
    ).sort_values(by="importance", ascending=False)

    for i in range(min(max_features, len(final_features))):
        feature_name = feature_importance_df["feature"].iloc[i]
        figfile_name = config.folder_name / f"s4_shap_dependence_{feature_name}.svg"

        plt.figure(figsize=(10, (9 / 16) * 10))
        shap.dependence_plot(
            ind=feature_name,
            shap_values=shap_values,
            features=data,
            feature_names=final_features,
            show=False,
        )
        plt.title(f"SHAP Dependence Plot: {feature_name}")
        plt.tight_layout()
        plt.savefig(figfile_name, dpi=config.dpi, format="svg")
        display(SVG(filename=str(figfile_name)))
        plt.close()


def plot_shap_feature_importance_ensemble(
    best_trial: Any,
    data: pd.DataFrame,
    final_features: List[str],
    config: Any,
    rng: Any,
    save_as: str = "s4_shap_feature_importance.svg",
) -> None:
    """
    Plot a SHAP feature importance bar plot for an ensemble.

    Parameters
    ----------
    best_trial : Any
        Optuna trial object.
    data : pd.DataFrame
        Feature matrix.
    final_features : List[str]
        List of selected features.
    config : Any
        Configuration object with output folder and DPI.
    rng : Any
        Random number generator.
    save_as : str, optional
        Output filename for the figure.
    """
    shap_values = np.concatenate(best_trial.user_attrs["all_shap_values"], axis=0)
    figfile_name = config.folder_name / save_as

    importance = np.abs(shap_values).mean(axis=0)
    feature_importance_df = pd.DataFrame(
        {
            "feature": final_features,
            "importance": importance,
        }
    ).sort_values(by="importance", ascending=True)

    plt.figure(figsize=(10, (9 / 16) * 10))
    shap.summary_plot(
        shap_values,
        data,
        feature_names=final_features,
        show=False,
        rng=rng,
        plot_type="bar",
    )

    for i, v in enumerate(feature_importance_df["importance"]):
        plt.text(v + 0.01, i, f"{v:.3f}", va="center")

    plt.title("SHAP Feature Importance Plot")
    plt.tight_layout()
    plt.savefig(figfile_name, dpi=config.dpi, format="svg")
    display(SVG(filename=str(figfile_name)))
    plt.close()


def visualize_ensemble_results(
    results: dict, final_features: List[str], config: Any, rng: Any
) -> None:
    """
    Generate and save all ensemble evaluation plots (ROC, SHAP summary, decision, dependence, and importance).

    Parameters
    ----------
    results : dict
        Dictionary containing ensemble results and SHAP values.
    final_features : List[str]
        List of selected features.
    config : Any
        Configuration object with output folder and DPI.
    rng : Any
        Random number generator.
    """
    fpr, tpr, roc_auc_val = results["fpr"], results["tpr"], results["roc_auc"]

    # ROC curve
    figfile = config.folder_name / "s4_ensemble_roc_curve.svg"
    plt.figure(figsize=(10, (9 / 16) * 10))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc_val:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Ensemble ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(figfile, dpi=config.dpi, format="svg")
    display(SVG(filename=str(figfile)))
    plt.close()

    # SHAP Summary Plot
    figfile = config.folder_name / "s4_shap_ensemble_summary.svg"
    plt.figure(figsize=(10, (9 / 16) * 10))
    shap.summary_plot(
        results["shap_values"],
        results["dmatrix"],
        feature_names=final_features,
        rng=rng,
        show=False,
    )
    plt.title("SHAP Ensemble Summary Plot")
    plt.tight_layout()
    plt.savefig(figfile, dpi=config.dpi, format="svg")
    display(SVG(filename=str(figfile)))
    plt.close()

    # SHAP Decision Plot
    figfile = config.folder_name / "s4_shap_ensemble_decision.svg"
    plt.figure(figsize=(10, (9 / 16) * 10))
    shap.decision_plot(
        base_value=results["explainer"].expected_value,
        shap_values=results["shap_values"],
        features=results["dmatrix"],
        feature_names=final_features,
        show=False,
    )
    plt.title("SHAP Ensemble Decision Plot")
    plt.tight_layout()
    plt.savefig(figfile, dpi=config.dpi, format="svg")
    display(SVG(filename=str(figfile)))
    plt.close()

    # SHAP Dependence & Importance Plots
    shap_vals = results["shap_values"]
    data = results["dmatrix"]
    feature_importance_df = pd.DataFrame(
        {
            "feature": final_features,
            "importance": np.abs(shap_vals).mean(axis=0),
        }
    ).sort_values(by="importance", ascending=False)

    for i in range(min(3, len(final_features))):
        feature_name = feature_importance_df["feature"].iloc[i]
        figfile = config.folder_name / f"s4_shap_ensemble_dependence_{feature_name}.svg"
        plt.figure(figsize=(10, (9 / 16) * 10))
        shap.dependence_plot(
            ind=feature_name,
            shap_values=shap_vals,
            features=data,
            feature_names=final_features,
            show=False,
        )
        plt.title(f"SHAP Ensemble Dependence Plot: {feature_name}")
        plt.tight_layout()
        plt.savefig(figfile, dpi=config.dpi, format="svg")
        display(SVG(filename=str(figfile)))
        plt.close()

    figfile = config.folder_name / "s4_shap_ensemble_feature_importance.svg"
    plt.figure(figsize=(10, (9 / 16) * 10))
    shap.summary_plot(
        shap_vals,
        data,
        feature_names=final_features,
        show=False,
        rng=rng,
        plot_type="bar",
    )
    for i, v in enumerate(
        feature_importance_df.sort_values(by="importance", ascending=True)["importance"]
    ):
        plt.text(v + 0.01, i, f"{v:.3f}", va="center")
    plt.title("SHAP Ensemble Feature Importance Plot")
    plt.tight_layout()
    plt.savefig(figfile, dpi=config.dpi, format="svg")
    display(SVG(filename=str(figfile)))
    plt.close()


def plot_class_distribution(class_counts: pd.Series, title: str = 'Class Distribution in Training Set') -> None:
    """
    Plot the class distribution as a bar chart.

    Parameters
    ----------
    class_counts : pd.Series
        Series with class counts (e.g., from y.value_counts()).
    title : str, optional
        Title for the plot.
    """
    print(f"Class distribution in training set: {class_counts.to_dict()}")
    plt.figure(figsize=(4, 2))
    class_counts.plot(kind='bar', color=['#4C72B0', '#55A868'])
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix_loocv(
    final_study: Any, y_true: np.ndarray, config: Any
) -> None:
    """
    Plot confusion matrix for the best trial in Stage 3 LOOCV.

    Parameters
    ----------
    final_study : Any
        Optuna study object containing best trials.
    y_true : np.ndarray
        True binary labels.
    config : Any
        Configuration object with output folder and DPI.
    """
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    
    # Get the best trial
    best_trial = final_study.best_trials[0] if final_study.best_trials else None
    if best_trial is None:
        print("No best trial found for confusion matrix.")
        return
    
    # Get binary predictions
    y_pred = best_trial.user_attrs.get("predictions_binary")
    if y_pred is None:
        print("No binary predictions found for confusion matrix.")
        return
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    figfile = config.folder_name / "s3_confusion_matrix.svg"
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    plt.title("Confusion Matrix - Stage 3 LOOCV (Best Trial)")
    plt.tight_layout()
    plt.savefig(figfile, dpi=config.dpi, format="svg")
    display(SVG(filename=str(figfile)))
    plt.close()


def plot_confusion_matrix_ensemble(
    best_trial: Any, config: Any
) -> None:
    """
    Plot confusion matrix for the best trial in Stage 4 Ensemble.

    Parameters
    ----------
    best_trial : Any
        Optuna trial object.
    config : Any
        Configuration object with output folder and DPI.
    """
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    
    # Get predictions and true labels from all folds
    valid_ys = best_trial.user_attrs.get("valid_ys")
    pred_labelss = best_trial.user_attrs.get("pred_labelss")
    
    if valid_ys is None or pred_labelss is None:
        print("No predictions found for confusion matrix.")
        return
    
    # Concatenate all folds
    y_true = np.concatenate(valid_ys)
    y_pred = np.concatenate(pred_labelss)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    figfile = config.folder_name / "s4_confusion_matrix.svg"
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    plt.title("Confusion Matrix - Stage 4 Ensemble (Best Trial)")
    plt.tight_layout()
    plt.savefig(figfile, dpi=config.dpi, format="svg")
    display(SVG(filename=str(figfile)))
    plt.close()
