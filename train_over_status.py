#!/usr/bin/env python3
"""Train classifiers to predict whether a trial is completed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


def _join_sequence(values: Iterable | float | str | dict | None) -> str:
    """Turn lists/iterables into a single string; leave scalars unchanged."""
    if values is None or (isinstance(values, float) and np.isnan(values)):
        return ""
    if isinstance(values, str):
        return values
    if isinstance(values, dict):
        return json.dumps(values, ensure_ascii=False)
    if isinstance(values, Iterable):
        try:
            return " | ".join(_join_sequence(v) for v in values if v)
        except TypeError:
            return str(values)
    return str(values)


def _parse_age(age_str: str | float | None) -> float:
    """Convert ClinicalTrials.gov age strings into years."""
    if age_str is None or (isinstance(age_str, float) and np.isnan(age_str)):
        return np.nan
    if not isinstance(age_str, str) or not age_str.strip():
        return np.nan
    parts = age_str.strip().split()
    try:
        val = float(parts[0])
    except (ValueError, IndexError):
        return np.nan
    unit = parts[1] if len(parts) > 1 else "Years"
    multipliers = {
        "Year": 1.0,
        "Years": 1.0,
        "Month": 1 / 12,
        "Months": 1 / 12,
        "Week": 1 / 52,
        "Weeks": 1 / 52,
        "Day": 1 / 365,
        "Days": 1 / 365,
    }
    return val * multipliers.get(unit, np.nan)


def load_dataset(path: Path) -> pd.DataFrame:
    """Read NDJSON lines file and derive features/labels."""
    df = pd.read_json(path, lines=True)
    if "overall_status" not in df.columns:
        raise ValueError("Expected column 'overall_status' in dataset.")

    df["label"] = df["overall_status"].str.lower().eq("completed").astype(int)

    seq_cols = [
        "conditions",
        "condition_mesh_terms",
        "keywords",
        "interventions",
        "collaborators",
        "locations",
    ]
    for col in seq_cols:
        if col in df.columns:
            df[col] = df[col].apply(_join_sequence)
        else:
            df[col] = ""

    df["minimum_age_years"] = df.get("minimum_age", np.nan).apply(_parse_age)
    df["maximum_age_years"] = df.get("maximum_age", np.nan).apply(_parse_age)
    df["enrollment_num"] = pd.to_numeric(df.get("enrollment", np.nan), errors="coerce")
    df["text_blob"] = (
        df.get("brief_title", "").fillna("")
        + " "
        + df.get("official_title", "").fillna("")
        + " "
        + df["conditions"].fillna("")
        + " "
        + df["keywords"].fillna("")
        + " "
        + df.get("why_stopped", "").fillna("")
    ).str.strip()
    return df


def build_pipeline(model) -> Pipeline:
    """Create preprocessing + estimator pipeline."""
    categorical = [
        "phase",
        "study_type",
        "lead_sponsor",
        "gender",
        "healthy_volunteers",
    ]
    numeric = [
        "minimum_age_years",
        "maximum_age_years",
        "enrollment_num",
    ]
    txt = "text_blob"

    preprocess = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
                    ]
                ),
                categorical,
            ),
            (
                "num",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            ("text", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2), txt),
        ],
        remainder="drop",
    )

    return Pipeline([("preprocess", preprocess), ("clf", model)])


def _resolve_test_size(
    requested: float | int,
    n_samples: int,
    n_classes: int,
    allow_stratify: bool,
) -> tuple[int | float, bool]:
    """Ensure the test split has enough samples for all classes."""
    if n_samples < 2:
        raise ValueError("Dataset must contain at least 2 samples to perform a split.")

    stratify = allow_stratify and n_classes > 1
    min_required = n_classes if stratify else 1

    if isinstance(requested, float):
        desired = int(np.ceil(requested * n_samples))
    else:
        desired = int(requested)

    desired = max(desired, min_required, 1)
    if desired >= n_samples:
        desired = n_samples - 1
        stratify = False  # not enough samples to keep stratification

    if isinstance(requested, float):
        resolved = desired / n_samples
    else:
        resolved = desired

    return resolved, stratify


def train_and_eval(data_path: Path, test_size: float | int, random_state: int, save_model: Path | None) -> None:
    df = load_dataset(data_path)
    X = df.drop(columns=["label"])
    y = df["label"]

    resolved_test_size, use_stratify = _resolve_test_size(test_size, len(X), y.nunique(), True)
    if resolved_test_size != test_size:
        print(
            f"Adjusted test_size from {test_size} to {resolved_test_size} to ensure enough samples per class."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=resolved_test_size,
        random_state=random_state,
        stratify=y if use_stratify else None,
    )

    models = {
        "logistic_regression": LogisticRegression(max_iter=600, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=600,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=800,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
        ),
    }

    trained_pipes: dict[str, Pipeline] = {}
    for name, estimator in models.items():
        pipe = build_pipeline(estimator)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]
        print(f"\n=== {name} ===")
        print(classification_report(y_test, preds, digits=4))
        try:
            print("ROC-AUC:", roc_auc_score(y_test, proba))
        except ValueError:
            print("ROC-AUC: cannot compute (needs both classes in y_test).")
        trained_pipes[name] = pipe

    ensemble = VotingClassifier(
        estimators=[(name, pipe.named_steps["clf"]) for name, pipe in trained_pipes.items()],
        voting="soft",
        weights=[2, 1, 2],  # tweak if validation suggests different importance
        n_jobs=-1,
    )
    ensemble_pipe = build_pipeline(ensemble)
    ensemble_pipe.fit(X_train, y_train)
    ens_preds = ensemble_pipe.predict(X_test)
    ens_proba = ensemble_pipe.predict_proba(X_test)[:, 1]
    print("\n=== soft ensemble ===")
    print(classification_report(y_test, ens_preds, digits=4))
    try:
        print("ROC-AUC:", roc_auc_score(y_test, ens_proba))
    except ValueError:
        print("ROC-AUC: cannot compute (needs both classes in y_test).")

    if save_model is not None:
        import joblib

        joblib.dump(ensemble_pipe, save_model)
        print(f"Saved ensemble pipeline to {save_model}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("/Users/leo/Desktop/520 Project 2/data/trials_summary_sample.ndjson"),
        help="Path to NDJSON dataset (default: sample file).",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split size fraction.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for splitting.")
    parser.add_argument(
        "--save-model",
        type=Path,
        default=None,
        help="Optional path to persist the fitted ensemble pipeline via joblib.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_and_eval(args.data, args.test_size, args.random_state, args.save_model)


if __name__ == "__main__":
    main()

