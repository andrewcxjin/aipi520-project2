#!/usr/bin/env python3
"""Train classifiers to predict whether a trial is completed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
plt.switch_backend("Agg")
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
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


def _save_classification_reports(reports: dict[str, dict], output_dir: Path) -> None:
    """Persist classification reports to JSON and CSV."""
    report_path = output_dir / "classification_reports.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(reports, fh, indent=2)

    rows: list[dict[str, float | str]] = []
    for model_name, report in reports.items():
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                rows.append(
                    {
                        "model": model_name,
                        "label": label,
                        "precision": metrics.get("precision"),
                        "recall": metrics.get("recall"),
                        "f1_score": metrics.get("f1-score"),
                        "support": metrics.get("support"),
                    }
                )
            else:
                rows.append(
                    {
                        "model": model_name,
                        "label": label,
                        "precision": metrics,
                        "recall": metrics,
                        "f1_score": metrics,
                        "support": None,
                    }
                )

    pd.DataFrame(rows).to_csv(output_dir / "classification_reports.csv", index=False)


def _plot_roc_curves(roc_data: dict[str, dict[str, list[float]]], output_dir: Path) -> None:
    """Plot ROC curves for all models that produced probabilities."""
    plt.figure(figsize=(8, 6))
    plotted = False
    for name, data in roc_data.items():
        if not data:
            continue
        plt.plot(data["fpr"], data["tpr"], label=f"{name} (AUC={data['roc_auc']:.3f})")
        plotted = True

    if not plotted:
        plt.close()
        return

    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curves.png", dpi=200)
    plt.close()


def _plot_confusion_matrix(cm: np.ndarray, output_dir: Path) -> None:
    """Plot confusion matrix for the ensemble predictions."""
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Not completed", "Completed"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Not completed", "Completed"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

    ax.set_title("Ensemble Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "ensemble_confusion_matrix.png", dpi=200)
    plt.close(fig)


def train_and_eval(
    data_path: Path,
    test_size: float | int,
    random_state: int,
    save_model: Path | None,
    output_dir: Path,
) -> None:
    df = load_dataset(data_path)
    X = df.drop(columns=["label"])
    y = df["label"]

    output_dir.mkdir(parents=True, exist_ok=True)

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
    reports: dict[str, dict] = {}
    roc_data: dict[str, dict[str, list[float]]] = {}

    for name, estimator in models.items():
        pipe = build_pipeline(estimator)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]
        print(f"\n=== {name} ===")
        report = classification_report(y_test, preds, digits=4, output_dict=True)
        print(classification_report(y_test, preds, digits=4))
        try:
            auc = roc_auc_score(y_test, proba)
            print("ROC-AUC:", auc)
            fpr, tpr, _ = roc_curve(y_test, proba)
            roc_data[name] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "roc_auc": auc,
            }
        except ValueError:
            print("ROC-AUC: cannot compute (needs both classes in y_test).")
            roc_data[name] = {}

        reports[name] = report
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
    ens_report = classification_report(y_test, ens_preds, digits=4, output_dict=True)
    reports["soft_ensemble"] = ens_report

    try:
        ens_auc = roc_auc_score(y_test, ens_proba)
        print("ROC-AUC:", ens_auc)
        fpr, tpr, _ = roc_curve(y_test, ens_proba)
        roc_data["soft_ensemble"] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "roc_auc": ens_auc,
        }
    except ValueError:
        print("ROC-AUC: cannot compute (needs both classes in y_test).")
        roc_data["soft_ensemble"] = {}

    cm = confusion_matrix(y_test, ens_preds)
    _save_classification_reports(reports, output_dir)
    _plot_roc_curves(roc_data, output_dir)
    _plot_confusion_matrix(cm, output_dir)

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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory where classification reports and plots will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_and_eval(args.data, args.test_size, args.random_state, args.save_model, args.output_dir)


if __name__ == "__main__":
    main()

