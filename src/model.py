"""Train and evaluate salary models on a time-based split."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "age", "G", "AB", "HR", "RBI", "SB", "BB", "SO",
    "AVG", "OBP", "SLG", "OPS", "HR_change",
    "seasons_played", "career_G", "career_HR", "career_H",
]
TEST_START_YEAR = 2012  # train on 1985-2011, test on 2012-2016


@dataclass
class Split:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    test_rows: pd.DataFrame


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["HR_change"] = df["HR_change"].fillna(0)
    # Model pay relative to the league average, on a log scale, so 1990 and 2015 are comparable
    df["log_salary_vs_league"] = np.log(df["salary_vs_league"])
    df["top_quartile"] = (df["salary_pct_rank"] >= 0.75).astype(int)
    return df.dropna(subset=FEATURES)


def time_split(df: pd.DataFrame, target: str) -> Split:
    train = df[df["salary_year"] < TEST_START_YEAR]
    test = df[df["salary_year"] >= TEST_START_YEAR]
    return Split(train[FEATURES], test[FEATURES], train[target], test[target], test)


def run_regression(df: pd.DataFrame) -> dict:
    s = time_split(df, "log_salary_vs_league")
    models = {
        "Linear regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Gradient boosting": GradientBoostingRegressor(random_state=42, n_estimators=300,
                                                       max_depth=3, learning_rate=0.05),
    }
    results, fitted = {}, {}
    for name, model in models.items():
        model.fit(s.X_train, s.y_train)
        pred = model.predict(s.X_test)
        # MAE in real dollars: convert predictions back from log(salary / league avg)
        pred_dollars = np.exp(pred) * s.test_rows["league_avg_salary"]
        results[name] = {
            "r2": round(r2_score(s.y_test, pred), 3),
            "mae_dollars": round(mean_absolute_error(s.test_rows["salary"], pred_dollars)),
        }
        fitted[name] = (model, pred)

    best_model, _ = fitted["Gradient boosting"]
    imp = permutation_importance(best_model, s.X_test, s.y_test, n_repeats=10, random_state=42)
    importance = (pd.Series(imp.importances_mean, index=FEATURES)
                  .sort_values(ascending=False))
    return {"metrics": results, "importance": importance, "split": s,
            "best_pred": fitted["Gradient boosting"][1]}


def run_classification(df: pd.DataFrame) -> dict:
    s = time_split(df, "top_quartile")
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    clf.fit(s.X_train, s.y_train)
    proba = clf.predict_proba(s.X_test)[:, 1]
    return {
        "accuracy": round(accuracy_score(s.y_test, proba >= 0.5), 3),
        "roc_auc": round(roc_auc_score(s.y_test, proba), 3),
        "baseline_accuracy": round(1 - s.y_test.mean(), 3),
    }
