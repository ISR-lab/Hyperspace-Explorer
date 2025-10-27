
import os
from tempfile import NamedTemporaryFile, mktemp
from typing import List

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import shap
from IPython.core.display_functions import display
from optuna.artifacts import upload_artifact, FileSystemArtifactStore, download_artifact
from scipy.stats import mode
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    roc_curve,
    auc,
)
from sklearn.model_selection import StratifiedKFold

from hyperspace_explorer.components.study_utils import create_dataset


class EnsembleStudyRunner:
    """
    Study runner for LightGBM ensemble hyperparameter optimization using Optuna.
    Provides methods for running ensemble studies, evaluating models, and reporting results.
    """

    def __init__(self, config, numerical_param_df, categorical_param_df):
        """
        Initialize the ensemble study runner.
        Args:
            config: Configuration object with experiment parameters.
            numerical_param_df: DataFrame of numerical parameter ranges.
            categorical_param_df: DataFrame of categorical parameter values.
        """
        self.config = config
        self.numerical_param_df = numerical_param_df
        self.categorical_param_df = categorical_param_df
        self.artifact_store = FileSystemArtifactStore(base_path=config.artifacts_folder)

    def expert_objective(self, trial, data, target):
        """
        Objective function for Optuna ensemble study. Trains LightGBM models with cross-validation,
        saves models as artifacts, and computes metrics.
        Args:
            trial: Optuna trial object.
            data: Feature DataFrame.
            target: Target array or Series.
        Returns:
            Tuple of (mean F1_Score_0, mean F1_Score_1, mean ROC_AUC)
        """
        # Define the parameter grid for Optuna to search
        param = self.config.set_ensemble_params(
            trial, self.numerical_param_df, self.categorical_param_df
        )

        # Initialize StratifiedKFold
        skf = StratifiedKFold(
            n_splits=self.config.cv_splits,
            shuffle=True,
            random_state=self.config.random_seed,
        )
        roc_curves = []
        all_shap_values = []
        valid_ys = []
        pred_labelss = []
        f1_0_scores = []
        f1_1_scores = []
        rocauc_scores = []
        accuracy_scores = []
        precisions_0 = []
        recalls_0 = []
        precisions_1 = []
        recalls_1 = []
        supports_0 = []
        supports_1 = []
        iterations = []
        artifact_ids = []

        for train_index, valid_index in skf.split(data, target):
            # Split the data into train and test set
            train_x, valid_x = data.iloc[train_index], data.iloc[valid_index]
            train_y, valid_y = target.iloc[train_index], target.iloc[valid_index]

            # Convert to Dataset
            dtrain = create_dataset(train_x, train_y)
            dvalid = create_dataset(valid_x, valid_y)

            # Train the first model and determine iterations
            fst = lgb.train(
                param,
                dtrain,
                num_boost_round=self.config.num_boost_round,
                valid_sets=[dvalid],
                callbacks=[
                    lgb.early_stopping(
                        stopping_rounds=self.config.early_stopping, verbose=False
                    )
                ],
            )

            # Train best model
            iterations.append(fst.best_iteration)
            bst = lgb.train(param, dtrain, num_boost_round=fst.best_iteration)

            # Create a temporary file path and close the file immediately
            with NamedTemporaryFile(suffix=".joblib", delete=False) as tmp_file:
                temp_model_path = tmp_file.name

            # Save the model after the file is closed
            joblib.dump(bst, temp_model_path)

            # Upload the model
            artifact_id = upload_artifact(
                artifact_store=self.artifact_store,
                file_path=temp_model_path,
                study_or_trial=trial,
            )

            # Remove the temp file
            os.remove(temp_model_path)

            artifact_ids.append(artifact_id)

            # Predict using the best iteration (early stopping point) -> probabilities for class 1
            preds = bst.predict(dvalid.data)  # shape (n_valid,); probabilities

            # Compute ROC curve (use probabilities)
            fpr, tpr, _ = roc_curve(valid_y, preds)
            roc_curves.append((fpr, tpr, auc(fpr, tpr)))

            # SHAP-based feature selection using TreeExplainer
            explainer = shap.TreeExplainer(bst)
            shap_values = explainer.shap_values(dvalid.data)

            # Use class 1 SHAP values if binary classification returns list
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            all_shap_values.append(shap_values)

            # Convert probabilities to binary labels via 0.5 for fold-level reporting
            pred_labels = (preds >= 0.5).astype(int)

            # Store per-fold arrays
            valid_ys.append(valid_y.values)
            pred_labelss.append(pred_labels)

            # Per-class F1
            f1_scores = f1_score(
                valid_y, pred_labels, average=None, zero_division=np.nan
            )
            f1_0_scores.append(f1_scores[0])
            f1_1_scores.append(f1_scores[1])

            # AUC from probabilities (not from hard labels)
            rocauc_scores.append(roc_auc_score(valid_y, preds))

            # Other metrics for reporting
            precisions_0.append(
                precision_score(valid_y, pred_labels, pos_label=0, zero_division=np.nan)
            )
            recalls_0.append(
                recall_score(valid_y, pred_labels, pos_label=0, zero_division=np.nan)
            )
            precisions_1.append(
                precision_score(valid_y, pred_labels, pos_label=1, zero_division=np.nan)
            )
            recalls_1.append(
                recall_score(valid_y, pred_labels, pos_label=1, zero_division=np.nan)
            )

            accuracy_scores.append(accuracy_score(valid_y, pred_labels))

            supports_0.append(np.sum(valid_y == 0))
            supports_1.append(np.sum(valid_y == 1))

        f1_0_mean = np.mean(f1_0_scores)
        f1_1_mean = np.mean(f1_1_scores)
        rocauc_mean = np.mean(rocauc_scores)

        # Store the metrics in trial.user_attrs
        trial.set_user_attr("valid_ys", valid_ys)
        trial.set_user_attr("pred_labelss", pred_labelss)
        trial.set_user_attr("f1_0_scores", f1_0_scores)
        trial.set_user_attr("f1_1_scores", f1_1_scores)
        trial.set_user_attr("rocauc_scores", rocauc_scores)
        trial.set_user_attr("precisions_0", precisions_0)
        trial.set_user_attr("recalls_0", recalls_0)
        trial.set_user_attr("precisions_1", precisions_1)
        trial.set_user_attr("recalls_1", recalls_1)
        trial.set_user_attr("accuracy_scores", accuracy_scores)
        trial.set_user_attr("supports_0", supports_0)
        trial.set_user_attr("supports_1", supports_1)
        trial.set_user_attr("roc_curves", roc_curves)
        trial.set_user_attr("all_shap_values", all_shap_values)
        trial.set_user_attr("iterations", iterations)
        trial.set_user_attr("artifact_ids", artifact_ids)

        return f1_0_mean, f1_1_mean, rocauc_mean

    def run_final_study(self, X, y, final_features: List[str]) -> dict:
        """
        Run the final Optuna ensemble study using selected features and summarize results.
        Args:
            X: Feature DataFrame.
            y: Target array or Series.
            final_features: List of selected features to use.
        Returns:
            Dictionary with study, best parameters, best trial DataFrame, and best trial object.
        """
        # Initialize study
        final_sampler = optuna.samplers.NSGAIISampler(seed=self.config.random_seed)
        final_study = optuna.create_study(
            directions=["maximize", "maximize", "maximize"],
            sampler=final_sampler,
        )

        final_study.optimize(
            lambda trial: self.expert_objective(trial, X[final_features], y),
            n_trials=self.config.n_final_trials,
            n_jobs=1,
            show_progress_bar=True,
        )

        final_good_trials = len(
            [
                t
                for t in final_study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
            ]
        )
        print("Number of finished trials: ", len(final_study.trials))
        if final_good_trials == 0:
            print("No completed trials. Increase number of trials.")
            return {}

        # Prepare best trial table
        final_score_columns = ["F1_Score_0", "F1_Score_1", "ROC_AUC"]
        final_df_best = pd.DataFrame(
            [(t.number, *t.values) for t in final_study.best_trials],
            columns=["Trial ID"] + final_score_columns,
        ).sort_values(
            by=["F1_Score_1", "ROC_AUC", "F1_Score_0"], ascending=[False, False, False]
        )
        display(final_df_best)

        best_trial_id = int(final_df_best.iloc[0]["Trial ID"])
        best_trial = final_study.trials[best_trial_id]

        best_params = self.config.get_base_params()
        best_params.update(best_trial.params)

        print(
            f"Best Trial ID: {best_trial_id}\n  Final Features: {final_features}\n  Params:"
        )
        for k, v in best_params.items():
            print(f"    {k}: {v}")

        # Extract user_attrs
        attrs = best_trial.user_attrs

        def mean_or_nan(lst):
            return np.nanmean(lst) if lst else float("nan")

        def int_mean(lst):
            return int(np.round(np.mean(lst))) if lst else 0

        avg_precision_0 = mean_or_nan(attrs.get("precisions_0", []))
        avg_recall_0 = mean_or_nan(attrs.get("recalls_0", []))
        avg_f1_0 = mean_or_nan(attrs.get("f1_0_scores", []))
        avg_precision_1 = mean_or_nan(attrs.get("precisions_1", []))
        avg_recall_1 = mean_or_nan(attrs.get("recalls_1", []))
        avg_f1_1 = mean_or_nan(attrs.get("f1_1_scores", []))
        avg_accuracy = mean_or_nan(attrs.get("accuracy_scores", []))
        avg_support_0 = int_mean(attrs.get("supports_0", []))
        avg_support_1 = int_mean(attrs.get("supports_1", []))
        total_support = avg_support_0 + avg_support_1

        avg_macro_precision = np.nanmean([avg_precision_0, avg_precision_1])
        avg_macro_recall = np.nanmean([avg_recall_0, avg_recall_1])
        avg_macro_f1 = np.nanmean([avg_f1_0, avg_f1_1])
        avg_weighted_precision = (
            avg_precision_0 * avg_support_0 + avg_precision_1 * avg_support_1
        ) / total_support
        avg_weighted_recall = (
            avg_recall_0 * avg_support_0 + avg_recall_1 * avg_support_1
        ) / total_support
        avg_weighted_f1 = (
            avg_f1_0 * avg_support_0 + avg_f1_1 * avg_support_1
        ) / total_support

        print("Estimators: ", attrs.get("iterations", []))
        print("\nAverage Classification Report Across Folds:")
        print(f"{'':>12}{'precision':>11}{'recall':>10}{'f1-score':>10}{'support':>10}")
        print(
            f"{'0':>12}{avg_precision_0:>11.3f}{avg_recall_0:>10.3f}{avg_f1_0:>10.3f}{avg_support_0:>10}"
        )
        print(
            f"{'1':>12}{avg_precision_1:>11.3f}{avg_recall_1:>10.3f}{avg_f1_1:>10.3f}{avg_support_1:>10}\n"
        )
        print(f"{'accuracy':>12}{avg_accuracy:>31.3f}{total_support:>10}")
        print(
            f"{'macro avg':>12}{avg_macro_precision:>11.3f}{avg_macro_recall:>10.3f}{avg_macro_f1:>10.3f}{total_support:>10}"
        )
        print(
            f"{'weighted avg':>12}{avg_weighted_precision:>11.3f}{avg_weighted_recall:>10.3f}{avg_weighted_f1:>10.3f}{total_support:>10}"
        )

        # Overall report from concatenated folds
        try:
            overall_report = classification_report(
                np.concatenate(attrs.get("valid_ys", [])),
                np.concatenate(attrs.get("pred_labelss", [])),
                zero_division=np.nan,
            )
            print("\n\nOverall Classification Report (Concatenated Folds):")
            print(overall_report)
        except Exception as e:
            print("Could not generate overall report:", e)

        return {
            "study": final_study,
            "best_params": best_params,
            "final_df_best": final_df_best,
            "best_trial": best_trial,
        }

    # --- NEW: probability-averaging helpers ---
    def ensemble_predict_proba_mean(self, models, data):
        """
        Return mean predicted probability for class 1 across models.
        """
        probas = [model.predict(data) for model in models]  # LightGBM Booster.predict -> prob class 1
        p_mean = np.mean(np.column_stack(probas), axis=1)
        return p_mean

    def ensemble_predict_labels_from_mean(self, models, data, threshold: float = 0.5):
        """
        Predict class labels from mean probability using a single threshold.
        """
        p_mean = self.ensemble_predict_proba_mean(models, data)
        return (p_mean >= threshold).astype(int)

    # Keeping majority vote for backward-compatibility (not used anymore)
    def ensemble_predict_majority_vote(self, models, data):
        all_predictions = np.array(
            [(model.predict(data) >= 0.5).astype(int) for model in models]
        )
        ensemble_predictions, _ = mode(all_predictions, axis=0, keepdims=False)
        return ensemble_predictions.flatten()

    def evaluate_ensemble(
        self, best_trial, holdout_X, holdout_y, final_features, create_dataset
    ):
        """
        Evaluate the ensemble on a holdout set, compute metrics, and generate SHAP explanations.
        Uses probability averaging (soft vote) with a single 0.5 threshold.
        """
        artifact_ids = best_trial.user_attrs.get("artifact_ids", [])
        dmatrix = create_dataset(holdout_X[final_features], holdout_y)

        # Load models
        ensemble_models = []
        for artifact_id in artifact_ids:
            model_path = mktemp(suffix=".joblib")
            download_artifact(
                artifact_store=self.artifact_store,
                artifact_id=artifact_id,
                file_path=model_path,
            )
            model = joblib.load(model_path)
            ensemble_models.append(model)
            os.remove(model_path)

        # --- Soft vote: mean probability ---
        p_mean = self.ensemble_predict_proba_mean(ensemble_models, dmatrix.data)
        ensemble_preds = (p_mean >= 0.5).astype(int)  # single threshold

        # Reports: use labels for class report, probabilities for ROC
        report = classification_report(holdout_y, ensemble_preds, zero_division=np.nan)
        fpr, tpr, _ = roc_curve(holdout_y, p_mean)
        roc_auc_val = auc(fpr, tpr)

        # SHAP explainer on the mean-probability ensemble
        def prediction_fn(data_array):
            # Return mean probability, not hard labels
            return self.ensemble_predict_proba_mean(ensemble_models, data_array)

        explainer = shap.KernelExplainer(prediction_fn, dmatrix.data)
        shap_values = explainer.shap_values(dmatrix.data)

        return {
            "models": ensemble_models,
            "preds": ensemble_preds,
            "probas": p_mean,
            "report": report,
            "fpr": fpr,
            "tpr": tpr,
            "roc_auc": roc_auc_val,
            "explainer": explainer,
            "shap_values": shap_values,
            "dmatrix": dmatrix.data,
        }
