import json
from pathlib import Path

import numpy as np


def evaluate(model, x_test: np.ndarray, y_test: np.ndarray, label_names: list[str], output_dir: Path) -> dict:
    from sklearn.metrics import classification_report, confusion_matrix

    probs = model.predict(x_test, verbose=0)
    y_pred = probs.argmax(axis=1)

    # Explicit `labels` so this never crashes even if a category ended up with zero
    # samples (e.g. an arXiv fetch failure) — classification_report/confusion_matrix
    # otherwise raise when the observed class count doesn't match len(target_names).
    all_label_indices = list(range(len(label_names)))
    report = classification_report(
        y_test, y_pred, labels=all_label_indices, target_names=label_names, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(y_test, y_pred, labels=all_label_indices).tolist()

    majority_class = np.bincount(y_test, minlength=len(label_names)).argmax()
    baseline_accuracy = float(np.mean(y_test == majority_class))

    results = {
        "accuracy": report["accuracy"],
        "majority_class_baseline_accuracy": baseline_accuracy,
        "per_class": {name: report[name] for name in label_names},
        "confusion_matrix": matrix,
        "label_order": label_names,
        "missing_classes": [label_names[i] for i in all_label_indices if i not in set(y_test.tolist())],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation.json").write_text(json.dumps(results, indent=2))
    return results
