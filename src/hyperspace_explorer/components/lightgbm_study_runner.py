import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import shap
from IPython.core.display_functions import display
from kneed import KneeLocator
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    classification_report,
    accuracy_score,
)
from sklearn.model_selection import train_test_split, LeaveOneOut

from hyperspace_explorer.components.study_utils import create_dataset


class LightGBMStudyRunner:
    """
    Study runner for LightGBM hyperparameter optimization using Optuna.
    Provides methods for running studies, analyzing results, and feature selection.
    """

    def __init__(self, config, X, y):
        """
        Initialize the study runner.

        Args:
            config: Configuration object with experiment parameters.
            X: Feature DataFrame.
            y: Target Series or array.
        """
        self.config = config
        self.X = X
        self.y = y
        self.study = None
        self.categorical_param_df = None
        self.numerical_param_df = None

    def create_study(self):
        """
        Create an Optuna study for multi-objective optimization.
        """
        sampler = optuna.samplers.QMCSampler(
            seed=self.config.random_seed, warn_independent_sampling=False
        )
        self.study = optuna.create_study(
            directions=["maximize", "maximize", "maximize"], sampler=sampler
        )

    def objective(self, trial):
        """
        Objective function for Optuna study. Trains LightGBM, selects features using SHAP,
        and returns F1 and ROC AUC scores.
    
        Args:
            trial: Optuna trial object.
    
        Returns:
            Tuple of (F1_Score_0, F1_Score_1, ROC_AUC)
        """
        param = self.config.get_base_params()
        param.update(
            {
                "subsample": trial.suggest_float("subsample", 0.2, 1.0),
                "learning_rate": trial.suggest_float("learning_rate", 0.1, 1.0),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 1.0),
                "max_depth": trial.suggest_int("max_depth", 3, 9, step=2),
                "min_child_weight": trial.suggest_int("min_child_weight", 2, 10),
                "grow_policy": trial.suggest_categorical(
                    "grow_policy", ["depthwise", "leaf"]
                ),
            }
        )
    
        train_x, valid_x, train_y, valid_y = train_test_split(
            self.X,
            self.y,
            test_size=self.config.test_split,
            random_state=self.config.random_seed,
            stratify=self.y,
        )
    
        dtrain = create_dataset(train_x, train_y)
        dvalid = create_dataset(valid_x, valid_y)
    
        # Train base model for SHAP
        bst = lgb.train(param, dtrain)
    
        # SHAP-based feature selection
        explainer = shap.TreeExplainer(bst)
        shap_values = explainer.shap_values(dvalid.data)
    
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # binary classification: class 1
    
        mean_shap = np.abs(shap_values).mean(axis=0)
    
        if np.all(mean_shap == 0):
            return np.nan, np.nan, np.nan
    
        sorted_shap = np.sort(mean_shap)[::-1]
        knee_locator = KneeLocator(
            range(len(sorted_shap)), sorted_shap, curve="convex", direction="decreasing"
        )
        optimal_threshold = sorted_shap[knee_locator.knee]
    
        sorted_pairs = sorted(
            zip(train_x.columns, mean_shap), key=lambda x: x[1], reverse=True
        )
        filtered_features = [
            feat
            for feat, imp in sorted_pairs
            if (
                imp > optimal_threshold
                if optimal_threshold <= 0
                else imp >= optimal_threshold
            )
        ]
    
        trial.set_user_attr("selected_features", filtered_features)
    
        dtrain_selected = create_dataset(train_x, train_y, filtered_features)
        dvalid_selected = create_dataset(valid_x, valid_y, filtered_features)
    
        # First round with early stopping
        fst_selected = lgb.train(
            param,
            dtrain_selected,
            num_boost_round=500,
            valid_sets=[dvalid_selected],
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=self.config.early_stopping, verbose=False
                )
            ],
        )
    
        # Retrain with best iteration
        bst_selected = lgb.train(
            param, dtrain_selected, num_boost_round=fst_selected.best_iteration
        )
    
        # Predictions (keep probs for AUC)
        pred_probs = bst_selected.predict(dvalid_selected.data)
        pred_labels = np.round(pred_probs)  # Still use labels for F1 if needed
    
        # Metrics (CHANGE HERE: Use pred_probs for roc_auc_score)
        f1_0 = f1_score(valid_y, pred_labels, pos_label=0, zero_division=np.nan)
        f1_1 = f1_score(valid_y, pred_labels, pos_label=1, zero_division=np.nan)
        roc_auc_val = roc_auc_score(valid_y, pred_probs)  # Fixed: on probabilities
    
        # Pruning check (now uses correct AUC)
        if np.any(np.array([f1_0, f1_1, roc_auc_val]) <= 0.5):
            return np.nan, np.nan, np.nan
    
        # Store for later analysis (probs already set)
        trial.set_user_attr("predictions", pred_labels.tolist())
        trial.set_user_attr("probas", pred_probs.tolist())
    
        return f1_0, f1_1, roc_auc_val

    def explore_hyperspace(self):
        """
        Run the Optuna study to explore the hyperparameter space.
        """
        if not self.study:
            raise RuntimeError(
                "Study has not been initialized. Call `create_study()` first."
            )
        self.study.optimize(
            self.objective,
            n_trials=self.config.n_trials,
            n_jobs=1,
            show_progress_bar=True,
        )

    def summarize_results(self):
        """
        Print summary of finished and successful trials.
        """
        trials = self.study.trials if self.study else []
        good_trials = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]

        print(f"📈 Number of finished trials: {len(trials)}")
        print(
            f"✅ Number of successful trials: {len(good_trials)}"
            if good_trials
            else "⚠️ No successful trials."
        )

    def summarize_scores(self) -> pd.DataFrame:
        """
        Return a DataFrame of the top-performing trials and their scores.

        Returns:
            DataFrame with trial IDs and scores.
        """
        if self.study is None:
            raise RuntimeError("Study has not been run yet.")

        score_columns = ["F1_Score_0", "F1_Score_1", "ROC_AUC"]
        best_values = [
            (trial.number, *trial.values) for trial in self.study.best_trials
        ]
        df_best = pd.DataFrame(best_values, columns=["Trial ID"] + score_columns)

        print("\n🏅 Top Trials by Objective Scores:")
        display(df_best)
        return df_best

    def show_selected_features(self) -> pd.DataFrame:
        """
        Display selected features for each best trial.

        Returns:
            DataFrame of selected features per trial.
        """
        if self.study is None:
            raise RuntimeError("Study has not been run yet.")

        df_features = (
            pd.DataFrame(
                [
                    trial.user_attrs["selected_features"]
                    for trial in self.study.best_trials
                ],
                index=[trial.number for trial in self.study.best_trials],
            )
            .reset_index()
            .rename(columns={"index": "Trial ID"})
        )

        print("\n🧠 Selected Features for Top Trials:")
        display(df_features)
        return df_features

    def analyze_parameters(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Return DataFrames summarizing categorical and numerical parameter distributions.

        Returns:
            Tuple of (categorical_param_df, numerical_param_df)
        """
        if self.study is None:
            raise RuntimeError("Study has not been run yet.")

        successful_trials = [
            (trial.number, trial.params, trial.user_attrs["selected_features"])
            for trial in self.study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
        ]

        trial_df = pd.DataFrame(
            successful_trials, columns=["Trial", "Params", "SelectedFeatures"]
        )
        trial_df = trial_df.join(pd.json_normalize(trial_df.pop("Params")))

        # Categorical parameters
        cat_cols = trial_df.drop(columns=["Trial", "SelectedFeatures"]).select_dtypes(
            exclude=["number"]
        )
        unique_values = {
            col: cat_cols[col].unique().tolist() for col in cat_cols.columns
        }
        max_len = max(len(v) for v in unique_values.values()) if unique_values else 0
        self.categorical_param_df = pd.DataFrame(
            {
                col: values + [np.nan] * (max_len - len(values))
                for col, values in unique_values.items()
            }
        )

        # Numerical parameters
        num_param_df = trial_df.drop(
            columns=["Trial", "SelectedFeatures"]
        ).select_dtypes(include=["number"])
        self.numerical_param_df = num_param_df.agg(["min", "max"])

        print("\n🔣 Unique Values for Categorical Parameters:")
        display(self.categorical_param_df)

        print("\n📈 Min/Max Range of Numerical Parameters:")
        display(self.numerical_param_df)

        return self.categorical_param_df, self.numerical_param_df

    def analyze_feature_rankings(self) -> tuple[pd.DataFrame, list[str]]:
        """
        Analyze and rank features across all successful trials based on frequency and position.

        Returns:
            Tuple of (ranking DataFrame, final list of selected features)
        """
        if self.study is None:
            raise RuntimeError("Study has not been run yet.")

        # Collect selected features per trial
        successful_trials = [
            (trial.number, trial.params, trial.user_attrs["selected_features"])
            for trial in self.study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
        ]

        if not successful_trials:
            print("⚠️ No successful trials to analyze.")
            return pd.DataFrame(), []

        trial_df = pd.DataFrame(
            successful_trials, columns=["Trial", "Params", "SelectedFeatures"]
        )
        trial_df = trial_df.join(pd.json_normalize(trial_df.pop("Params")))

        # Expand features into long format
        trial_df_long = (
            trial_df.explode("SelectedFeatures")
            .rename(columns={"SelectedFeatures": "Feature"})
            .dropna()
        )

        # Compute Frequency, RankSum, and RankSum_Per_Frequency
        feat_stats_df = (
            trial_df_long.assign(RankSum=trial_df_long.groupby("Trial").cumcount() + 1)
            .groupby("Feature")
            .agg(Frequency=("Feature", "count"), RankSum=("RankSum", "sum"))
            .assign(RankSum_Per_Frequency=lambda x: x["RankSum"] / x["Frequency"])
            .sort_values(by="RankSum_Per_Frequency")
            .reset_index()
        )

        print("\n📊 Feature Selection Stats Across Trials:")
        display(feat_stats_df)

        # Count features per trial to calculate cumulative sum
        feature_counts_df = (
            trial_df_long.groupby("Trial").size().value_counts().sort_index()
        )
        cumulative_sums = feature_counts_df.cumsum()
        threshold = self.config.feat_count_cov * feature_counts_df.sum()
        rspf_cutoff = cumulative_sums.index[cumulative_sums.searchsorted(threshold)]

        print(
            f"\n📐 RankSum_Per_Frequency Cutoff (coverage={self.config.feat_count_cov:.0%}):",
            rspf_cutoff,
        )

        # Select features under the cutoff
        filtered_df_by_rspf = feat_stats_df[
            feat_stats_df["RankSum_Per_Frequency"] <= rspf_cutoff
        ].reset_index(drop=True)

        print("\n🏁 Final Feature Set (after cutoff):")
        display(filtered_df_by_rspf)

        final_features = filtered_df_by_rspf["Feature"].tolist()
        print("🧠 Final Selected Features:", final_features)

        return filtered_df_by_rspf, final_features

    def analyze_feature_frequencies(
        self,
        trial_df_long: pd.DataFrame,
        feat_stats_df: pd.DataFrame,
        final_features: list[str],
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series, float]:
        """
        Given exploded trial features and ranked feature stats,
        compute frequencies, cumulative stats, and % coverage of final features.

        Args:
            trial_df_long: DataFrame with exploded features per trial.
            feat_stats_df: DataFrame with feature statistics.
            final_features: List of selected features.

        Returns:
            Tuple of (frequency_df_sorted, feature_amount_freq, cumulative_percentages, selected_features_coverage_pct)
        """
        feature_counts = trial_df_long.groupby("Trial").size()
        feature_amount_freq = feature_counts.value_counts().sort_index()
        cumulative_percentages = (
            feature_amount_freq.cumsum() / feature_amount_freq.sum() * 100
        )

        print("\n📊 Frequency of number of features selected per trial:")
        print(feature_amount_freq)
        print("\n📈 Cumulative % of feature set sizes across trials:")
        print(cumulative_percentages)

        frequency_df_sorted = feat_stats_df.assign(
            Selected=feat_stats_df["Feature"].isin(final_features)
        ).sort_values(by=["Selected", "Frequency"], ascending=[False, False])

        frequency_df_sorted["Cumulative_Frequency"] = frequency_df_sorted[
            "Frequency"
        ].cumsum()
        frequency_df_sorted["Cumulative_Percentage"] = (
            frequency_df_sorted["Cumulative_Frequency"]
            / frequency_df_sorted["Frequency"].sum()
            * 100
        )

        selected_coverage_pct = (
            frequency_df_sorted.loc[frequency_df_sorted["Selected"], "Frequency"].sum()
            / frequency_df_sorted["Frequency"].sum()
            * 100
        )

        print(
            f"\n✅ Coverage of selected features across all feature picks: {selected_coverage_pct:.2f}%"
        )

        print("\n📋 Top features sorted by frequency and selection status:")
        display(frequency_df_sorted)

        return (
            frequency_df_sorted,
            feature_amount_freq,
            cumulative_percentages,
            selected_coverage_pct,
        )

    def _final_objective(self, trial, data, target):
        """
        Objective function for the final study using LOOCV.

        Args:
            trial: Optuna trial object.
            data: Feature DataFrame.
            target: Target array or Series.

        Returns:
            Tuple of (F1_Score_0, F1_Score_1, ROC_AUC)
        """
        # Define the parameter grid for Optuna to search
        # TODO should we do this via self --> numerical_param_df, categorical_param_df?
        param = self.config.set_loocv_params(
            trial, self.numerical_param_df, self.categorical_param_df
        )

        # Initialize LeaveOneOut
        loo = LeaveOneOut()
        predictions_proba = np.zeros(len(target))  # Store probabilities for ROC
        predictions_binary = np.zeros(len(target))  # Store binary predictions for F1
        all_shap_values = []
        iterations = []

        for train_index, test_index in loo.split(data):
            X_train, X_test = data.iloc[train_index], data.iloc[test_index]
            y_train, y_test = target.iloc[train_index], target.iloc[test_index]

            dtrain = create_dataset(X_train, y_train)
            dtest = create_dataset(X_test, y_test)

            # Train the pre model and determine iterations
            premodel = lgb.train(
                param,
                dtrain,
                num_boost_round=500,
                valid_sets=[dtest],
                callbacks=[
                    lgb.early_stopping(
                        stopping_rounds=self.config.early_stopping, verbose=False
                    )
                ],
            )

            # Train final model with best iterations
            iterations.append(premodel.best_iteration)
            model = lgb.train(param, dtrain, num_boost_round=premodel.best_iteration)

            # Predict using the best iteration (early stopping point)
            # Store probability for ROC curve generation
            predictions_proba[test_index[0]] = model.predict(dtest.data)[0]
            # Store binary prediction for F1 score calculation
            predictions_binary[test_index[0]] = predictions_proba[test_index[0]] > 0.5

            # SHAP-based feature selection using TreeExplainer
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(dtest.data)

            # Use class 1 SHAP values if binary classification
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            all_shap_values.append(shap_values)

        # Calculate metrics
        f1_scores = f1_score(target, predictions_binary, average=None, zero_division=np.nan)
        f1_class_0 = f1_scores[0]
        f1_class_1 = f1_scores[1]
        auc = roc_auc_score(target, predictions_proba)

        # Store the metrics in trial.user_attrs
        trial.set_user_attr("predictions", predictions_proba)  # Store probabilities for ROC
        trial.set_user_attr("predictions_binary", predictions_binary)  # Store binary for confusion matrix
        trial.set_user_attr("all_shap_values", all_shap_values)
        trial.set_user_attr("iterations", iterations)

        # Handle return of trial
        return (
            (np.nan, np.nan, np.nan)
            if any(x <= 0.5 for x in [f1_class_0, f1_class_1, auc])
            else (f1_class_0, f1_class_1, auc)
        )

    def explore_final_hyperspace(self, data, target):
        """
        Run the final Optuna study using LOOCV and multi-objective optimization.

        Args:
            data: Feature DataFrame.
            target: Target array or Series.

        Returns:
            The final Optuna study object.
        """
        final_sampler = optuna.samplers.NSGAIISampler(seed=self.config.random_seed)
        self.final_study = optuna.create_study(
            directions=["maximize", "maximize", "maximize"], sampler=final_sampler
        )

        self.final_study.optimize(
            lambda trial: self._final_objective(trial, data, target),
            n_trials=self.config.n_final_trials,
            n_jobs=1,
            show_progress_bar=True,
        )

        good_trials = [
            t
            for t in self.final_study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]

        print("🧪 Final Study Results")
        print("📊 Total trials run:", len(self.final_study.trials))
        if good_trials:
            print("✅ Successful trials:", len(good_trials))
        else:
            print("⚠️ No completed trials — try increasing N_FINAL_TRIALS")

        return self.final_study

    def get_final_scores_df(self, final_study) -> pd.DataFrame:
        """
        Return sorted performance table for final study best trials.

        Args:
            final_study: Optuna study object.

        Returns:
            DataFrame of sorted best trials.
        """
        final_score_columns = ["F1_Score_0", "F1_Score_1", "ROC_AUC"]
        df = pd.DataFrame(
            [(trial.number, *trial.values) for trial in final_study.best_trials],
            columns=["Trial ID"] + final_score_columns,
        )
        df_sorted = df.sort_values(
            by=["F1_Score_1", "ROC_AUC", "F1_Score_0"], ascending=[False, False, False]
        )
        print("📋 Final Best Trials (sorted):")
        display(df_sorted)
        return df_sorted

    def get_best_final_trial_params(
        self, final_study, df_best: pd.DataFrame
    ) -> tuple[dict, optuna.trial.FrozenTrial]:
        """
        Returns full param set of top-scoring final trial.

        Args:
            final_study: Optuna study object.
            df_best: DataFrame of best trials.

        Returns:
            Tuple of (full_params dict, best_trial object)
        """
        best_trial_id = int(df_best.iloc[0]["Trial ID"])
        best_trial = final_study.trials[best_trial_id]

        full_params = self.config.get_base_params()
        full_params.update(best_trial.params)

        print(f"\n🏆 Best Final Trial ID: {best_trial_id}")
        print("⚙️ Best Final Params:")
        for key, value in full_params.items():
            print(f"    {key}: {value}")

        return full_params, best_trial

    def summarize_final_classification_report(self, final_study, true_labels) -> dict:
        """
        Aggregates classification reports across best trials.

        Args:
            final_study: Optuna study object.
            true_labels: Ground truth labels.

        Returns:
            Dictionary with averaged classification metrics.
        """
        prec_0, rec_0, f1_0, supp_0 = [], [], [], []
        prec_1, rec_1, f1_1, supp_1 = [], [], [], []
        accuracy_scores = []

        for trial in final_study.best_trials:
            y_pred = trial.user_attrs.get("predictions_binary")  # Use binary predictions for classification report
            if y_pred is not None:
                report = classification_report(
                    true_labels, y_pred, output_dict=True, zero_division=np.nan
                )
                accuracy_scores.append(accuracy_score(true_labels, y_pred))
                for cls, prec, rec, f1, supp in [
                    ("0", prec_0, rec_0, f1_0, supp_0),
                    ("1", prec_1, rec_1, f1_1, supp_1),
                ]:
                    prec.append(report[cls]["precision"])
                    rec.append(report[cls]["recall"])
                    f1.append(report[cls]["f1-score"])
                    supp.append(report[cls]["support"])

        avg = {
            "0": {
                "precision": np.nanmean(prec_0),
                "recall": np.nanmean(rec_0),
                "f1-score": np.nanmean(f1_0),
                "support": int(np.round(np.mean(supp_0))),
            },
            "1": {
                "precision": np.nanmean(prec_1),
                "recall": np.nanmean(rec_1),
                "f1-score": np.nanmean(f1_1),
                "support": int(np.round(np.mean(supp_1))),
            },
        }

        total_support = avg["0"]["support"] + avg["1"]["support"]
        avg_accuracy = np.nanmean(accuracy_scores)

        avg["macro avg"] = {
            "precision": np.nanmean([avg["0"]["precision"], avg["1"]["precision"]]),
            "recall": np.nanmean([avg["0"]["recall"], avg["1"]["recall"]]),
            "f1-score": np.nanmean([avg["0"]["f1-score"], avg["1"]["f1-score"]]),
            "support": total_support,
        }

        avg["weighted avg"] = {
            "precision": (
                avg["0"]["precision"] * avg["0"]["support"]
                + avg["1"]["precision"] * avg["1"]["support"]
            )
            / total_support,
            "recall": (
                avg["0"]["recall"] * avg["0"]["support"]
                + avg["1"]["recall"] * avg["1"]["support"]
            )
            / total_support,
            "f1-score": (
                avg["0"]["f1-score"] * avg["0"]["support"]
                + avg["1"]["f1-score"] * avg["1"]["support"]
            )
            / total_support,
            "support": total_support,
        }

        print("\n📊 Average Classification Report Across Best Trials:")
        print(
            f"{'':>12}{'precision':>11}{'recall':>10}{'f1-score':>10}{'support':>10}\n"
        )
        for label in ["0", "1", "accuracy", "macro avg", "weighted avg"]:
            if label == "accuracy":
                print(f"{label:>12}{avg_accuracy:>31.3f}{total_support:>10}")
            else:
                row = avg[label]
                print(
                    f"{label:>12}{row['precision']:>11.3f}{row['recall']:>10.3f}{row['f1-score']:>10.3f}{row['support']:>10}"
                )

        return avg

    def summarize_combined_report(self, final_study, true_labels):
        """
        Flattens all predictions from best trials and prints overall report.

        Args:
            final_study: Optuna study object.
            true_labels: Ground truth labels.
        """
        all_predictions, all_y = [], []

        for trial in final_study.best_trials:
            preds = trial.user_attrs.get("predictions_binary")  # Use binary predictions for classification report
            if preds is not None:
                all_predictions.extend(preds)
                all_y.extend(true_labels)

        report = classification_report(all_y, all_predictions, zero_division=np.nan)
        print("\n📈 Overall Classification Report Across All Best Trials:")
        print(report)
