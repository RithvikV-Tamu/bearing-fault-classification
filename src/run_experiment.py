"""Train and evaluate all models, then save metrics and confusion matrices.

Main evaluation: leave-one-load-out. Each fold trains on three motor loads and
tests on the fourth, so windows from one recording are never in both sets.
Stricter check: leave-one-fault-size-and-load-out, so the damaged bearing in
the test set is also never seen in training.
A random split of overlapping windows is not used, because neighboring
windows from the same recording share samples and would leak into the test set.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score

from cwru_files import CLASS_NAMES
from features import fft_spectrum_features, handcrafted_features
from load_data import load_all_windows
from models import (
    predict_logistic_regression,
    predict_neural_network,
    train_logistic_regression,
    train_neural_network,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

MODELS = [
    ("Logistic regression, 4 hand-made features", "handcrafted",
     train_logistic_regression, predict_logistic_regression),
    ("Logistic regression, FFT spectrum", "spectrum",
     train_logistic_regression, predict_logistic_regression),
    ("Neural network (PyTorch MLP), FFT spectrum", "spectrum",
     train_neural_network, predict_neural_network),
]


def leave_one_load_out_folds(dataset):
    folds = []
    for held_out_load in sorted(set(dataset["motor_loads"])):
        is_test = dataset["motor_loads"] == held_out_load
        folds.append((f"test on {held_out_load} HP", np.where(~is_test)[0], np.where(is_test)[0]))
    return folds


def leave_one_size_and_load_out_folds(dataset):
    """Stricter check. Each CWRU fault size is one damaged bearing tested at all
    four loads, so leave-one-load-out still trains on the same damaged bearing.
    Here a fold tests one fault size at one load (plus the normal recording at
    that load) and trains only on other sizes at other loads."""
    folds = []
    fault_sizes = sorted(size for size in set(dataset["fault_diameters"]) if size > 0)
    is_normal = dataset["fault_diameters"] == 0
    for held_out_size in fault_sizes:
        for held_out_load in sorted(set(dataset["motor_loads"])):
            is_held_out_load = dataset["motor_loads"] == held_out_load
            is_held_out_size = dataset["fault_diameters"] == held_out_size
            is_test = is_held_out_load & (is_held_out_size | is_normal)
            is_train = ~is_held_out_load & (~is_held_out_size | is_normal)
            folds.append((
                f"test on {held_out_size} in at {held_out_load} HP",
                np.where(is_train)[0],
                np.where(is_test)[0],
            ))
    return folds


def evaluate_model(features, labels, folds, train_function, predict_function):
    fold_results = []
    all_true_labels = []
    all_predicted_labels = []
    for fold_name, train_indices, test_indices in folds:
        model = train_function(features[train_indices], labels[train_indices])
        predictions = predict_function(model, features[test_indices])
        true_labels = labels[test_indices]
        fold_results.append({
            "fold": fold_name,
            "test_windows": int(len(test_indices)),
            "accuracy": float(accuracy_score(true_labels, predictions)),
            "macro_f1": float(f1_score(true_labels, predictions, average="macro")),
        })
        all_true_labels.append(true_labels)
        all_predicted_labels.append(predictions)

    all_true_labels = np.concatenate(all_true_labels)
    all_predicted_labels = np.concatenate(all_predicted_labels)
    fold_accuracies = [fold["accuracy"] for fold in fold_results]
    fold_macro_f1s = [fold["macro_f1"] for fold in fold_results]
    return {
        "pooled_accuracy": float(accuracy_score(all_true_labels, all_predicted_labels)),
        "pooled_macro_f1": float(f1_score(all_true_labels, all_predicted_labels, average="macro")),
        "mean_fold_accuracy": float(np.mean(fold_accuracies)),
        "min_fold_accuracy": float(np.min(fold_accuracies)),
        "mean_fold_macro_f1": float(np.mean(fold_macro_f1s)),
        "min_fold_macro_f1": float(np.min(fold_macro_f1s)),
        "folds": fold_results,
        "confusion_matrix": confusion_matrix(
            all_true_labels, all_predicted_labels, labels=range(len(CLASS_NAMES))
        ).tolist(),
    }


def save_confusion_matrix_plot(matrix, title, file_path):
    figure, axis = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(np.array(matrix), display_labels=CLASS_NAMES).plot(
        ax=axis, cmap="Blues", values_format="d", colorbar=False
    )
    axis.set_title(title, fontsize=10)
    figure.tight_layout()
    figure.savefig(file_path, dpi=150)
    plt.close(figure)


def percent(value):
    return f"{100 * value:.1f}%"


def write_markdown_summary(dataset, all_results, file_path):
    lines = ["# Results", ""]
    lines.append(f"Windows: {len(dataset['labels'])} from {len(set(dataset['file_numbers']))} recordings.")
    for class_index, class_name in enumerate(CLASS_NAMES):
        is_class = dataset["labels"] == class_index
        recording_count = len(set(dataset["file_numbers"][is_class]))
        lines.append(f"- {class_name}: {is_class.sum()} windows from {recording_count} recordings")
    lines.append("")

    for split_name, split_results in all_results.items():
        lines.append(f"## {split_name}")
        lines.append("")
        lines.append("| Model | Accuracy (pooled) | Macro F1 (pooled) | Worst-fold accuracy | Worst-fold macro F1 |")
        lines.append("|---|---|---|---|---|")
        for model_name, result in split_results.items():
            lines.append(
                f"| {model_name} | {percent(result['pooled_accuracy'])} | {result['pooled_macro_f1']:.3f} "
                f"| {percent(result['min_fold_accuracy'])} | {result['min_fold_macro_f1']:.3f} |"
            )
        lines.append("")
        lines.append("Per fold (accuracy / macro F1):")
        lines.append("")
        for model_name, result in split_results.items():
            fold_text = ", ".join(
                f"{fold['fold']}: {percent(fold['accuracy'])} / {fold['macro_f1']:.3f}"
                for fold in result["folds"]
            )
            lines.append(f"- {model_name}: {fold_text}")
        lines.append("")
        lines.append("Confusion matrices (rows = true class, columns = predicted; order "
                     + ", ".join(CLASS_NAMES) + "):")
        lines.append("")
        for model_name, result in split_results.items():
            lines.append(f"- {model_name}: `{result['confusion_matrix']}`")
        lines.append("")

    file_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    print("Loading recordings and cutting windows...")
    dataset = load_all_windows()
    feature_sets = {
        "handcrafted": handcrafted_features(dataset["windows"]),
        "spectrum": fft_spectrum_features(dataset["windows"]),
    }
    labels = dataset["labels"]
    print(f"{len(labels)} windows, spectrum features: {feature_sets['spectrum'].shape[1]}")

    split_schemes = {
        "Leave-one-load-out": leave_one_load_out_folds(dataset),
        "Leave-one-fault-size-and-load-out (stricter)": leave_one_size_and_load_out_folds(dataset),
    }

    all_results = {}
    for split_name, folds in split_schemes.items():
        all_results[split_name] = {}
        for model_name, feature_key, train_function, predict_function in MODELS:
            print(f"{split_name} | {model_name}")
            result = evaluate_model(feature_sets[feature_key], labels, folds, train_function, predict_function)
            all_results[split_name][model_name] = result
            print(f"    accuracy {percent(result['pooled_accuracy'])}, macro F1 {result['pooled_macro_f1']:.3f}")

    model_file_tags = ["lr_handcrafted", "lr_spectrum", "nn_spectrum"]
    split_file_tags = ["load_out", "size_and_load_out"]
    for split_name, split_tag in zip(split_schemes, split_file_tags):
        for (model_name, *_), model_tag in zip(MODELS, model_file_tags):
            result = all_results[split_name][model_name]
            title = f"{model_name}\n{split_name}, accuracy {percent(result['pooled_accuracy'])}"
            file_path = RESULTS_DIR / f"confusion_{model_tag}_{split_tag}.png"
            save_confusion_matrix_plot(result["confusion_matrix"], title, file_path)

    with open(RESULTS_DIR / "metrics.json", "w", encoding="utf-8") as metrics_file:
        json.dump(all_results, metrics_file, indent=2)
    write_markdown_summary(dataset, all_results, RESULTS_DIR / "results.md")
    print(f"Saved results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
