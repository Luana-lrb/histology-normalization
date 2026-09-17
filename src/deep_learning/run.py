from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import csv
import math

from config import (
    SPLITS_DIR,
    RAW_DATASET_DIR,
    PROCESSED_DATASET_DIR,
    RESULTS_DIR,
    DEVICE,
    BATCH_SIZE,
    NUM_WORKERS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EPOCHS,
    SEEDS,
    PATIENCE,
    CLASS_NAMES,
    TOP_K_NORMALIZATIONS
)

from experiments import EXPERIMENT_0, EXPERIMENT_1, create_experiment_2

from dataset import HistologyDataset

from transforms import (
    create_train_transform,
    validation_transform
)

from model import create_model

from train import train_model

from evaluate import (
    load_model,
    create_evaluation_loader,
    predict,
    evaluate_model
)

from metrics import save_classification_metrics

from utils.seed import set_seed

from utils.plots import (
    plot_confusion_matrix,
    plot_training_history,
    plot_training_accuracy,
    plot_roc_curve
)


def get_dataset_dir(dataset_type, reference, normalization):

    if dataset_type == "raw":
        return RAW_DATASET_DIR

    if normalization == "multitarget_macenko":
        return PROCESSED_DATASET_DIR / "multitarget_macenko"

    return (
        PROCESSED_DATASET_DIR
        / reference
        / normalization
    )


def get_experiment_name(dataset_type, reference, normalization):

    if dataset_type == "raw":
        return "raw"

    if normalization == "multitarget_macenko":
        return "multitarget_macenko"

    return f"{normalization}_{reference}"


def train_experiment(
    dataset_type,
    reference,
    normalization,
    experiment_dir,
    seed,
    use_augmentation
):

    set_seed(seed)
    
    train_transform = create_train_transform(use_augmentation)

    dataset_dir = get_dataset_dir(
        dataset_type,
        reference,
        normalization
    )

    train_dataset = HistologyDataset(
        csv_file=SPLITS_DIR / "train.csv",
        root_dir=dataset_dir,
        transform=train_transform
    )

    validation_dataset = HistologyDataset(
        csv_file=SPLITS_DIR / "validation.csv",
        root_dir=dataset_dir,
        transform=validation_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    model = create_model().to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=3
    )

    save_path = experiment_dir / "best_model.pth"

    history = train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
        epochs=EPOCHS,
        patience=PATIENCE,
        save_path=save_path
    )

    plot_training_history(
        history,
        experiment_dir / "training_loss.png"
    )

    plot_training_accuracy(
        history,
        experiment_dir / "training_accuracy.png"
    )

    model = load_model(
        save_path,
        DEVICE
    )

    validation_labels, validation_predictions, validation_probabilities = predict(
        model,
        validation_loader,
        DEVICE
    )

    validation_metrics, _ = evaluate_model(
        validation_labels,
        validation_predictions,
        validation_probabilities
    )

    return {
        "dataset_type": dataset_type,
        "reference": reference,
        "normalization": normalization,
        "augmentation": use_augmentation,
        "experiment_dir": experiment_dir,
        "model_path": save_path,
        "seed": seed,
        "validation_metrics": validation_metrics
    }


def evaluate_test(
    experiment,
    test_loader
):

    model = load_model(
        experiment["model_path"],
        DEVICE
    )

    labels, predictions, probabilities = predict(
        model,
        test_loader,
        DEVICE
    )

    metrics, confusion = evaluate_model(
        labels,
        predictions,
        probabilities
    )

    experiment_dir = experiment["experiment_dir"]

    save_classification_metrics(
        metrics,
        experiment_dir / "classification_metrics.csv"
    )

    plot_confusion_matrix(
        confusion,
        CLASS_NAMES,
        experiment_dir / "confusion_matrix.png"
    )

    plot_roc_curve(
        labels,
        probabilities,
        CLASS_NAMES,
        experiment_dir / "roc_curve.png"
    )

    return metrics


def save_all_results(results, save_path):

    fieldnames = [
        "experiment",
        "dataset_type",
        "reference",
        "normalization",
        "seed",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc"
    ]

    with open(save_path, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)
        
def save_classification_summary(results, save_path):

    grouped_results = {}

    for result in results:

        key = (
            result["experiment"],
            result["dataset_type"],
            result["reference"],
            result["normalization"]
        )

        if key not in grouped_results:
            grouped_results[key] = []

        grouped_results[key].append(result)

    fieldnames = [
        "experiment",
        "dataset_type",
        "reference",
        "normalization",
        "accuracy_mean",
        "accuracy_std",
        "precision_mean",
        "precision_std",
        "recall_mean",
        "recall_std",
        "f1_mean",
        "f1_std",
        "auc_mean",
        "auc_std"
    ]

    with open(save_path, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for key, experiment_results in grouped_results.items():

            experiment, dataset_type, reference, normalization = key

            row = {
                "experiment": experiment,
                "dataset_type": dataset_type,
                "reference": reference,
                "normalization": normalization
            }

            for metric in [
                "accuracy",
                "precision",
                "recall",
                "f1",
                "auc"
            ]:

                values = [
                    result[metric]
                    for result in experiment_results
                ]

                mean = sum(values) / len(values)

                if len(values) > 1:
                    variance = sum(
                        (value - mean) ** 2
                        for value in values
                    ) / (len(values) - 1)

                    std = math.sqrt(variance)

                else:
                    std = 0.0

                row[f"{metric}_mean"] = mean
                row[f"{metric}_std"] = std

            writer.writerow(row)
            
def calculate_best_augmentation(results):

    validation_f1 = {}

    for result in results:

        augmentation = result["augmentation"]

        if augmentation not in validation_f1:
            validation_f1[augmentation] = []

        validation_f1[augmentation].append(
            result["validation_metrics"]["f1"]
        )

    mean_f1 = {
        augmentation: sum(scores) / len(scores)
        for augmentation, scores in validation_f1.items()
    }

    return max(
        mean_f1,
        key=mean_f1.get
    )


def calculate_best_normalizations(results):

    validation_f1 = {}

    for result in results:

        if result["dataset_type"] != "processed":
            continue

        normalization = result["normalization"]

        if normalization not in validation_f1:
            validation_f1[normalization] = []

        validation_f1[normalization].append(
            result["validation_metrics"]["f1"]
        )

    mean_f1 = {
        normalization: sum(scores) / len(scores)
        for normalization, scores in validation_f1.items()
    }

    ranked_normalizations = sorted(
        mean_f1,
        key=mean_f1.get,
        reverse=True
    )

    return ranked_normalizations[:TOP_K_NORMALIZATIONS]


def main():

    results_dir = RESULTS_DIR / "deep_learning"

    results_dir.mkdir(
        parents=True,
        exist_ok=True
    )
    
    print("\n========================================")
    print("EXPERIMENTO 0")
    print("========================================\n")

    
    experiment_0_results = []

    for augmentation_name, use_augmentation in EXPERIMENT_0:

        for seed in SEEDS:

            experiment_dir = (
                results_dir
                / "experiment_0"
                / augmentation_name
                / f"seed_{seed}"
            )

            result = train_experiment(
                dataset_type="raw",
                reference=None,
                normalization=None,
                experiment_dir=experiment_dir,
                seed=seed,
                use_augmentation=use_augmentation
            )

            experiment_0_results.append(result)
            
    best_augmentation = calculate_best_augmentation(
        experiment_0_results
    )
    
    augmentation_name = (
        "com augmentation"
        if best_augmentation
        else "sem augmentation"
    )

    print("\n========================================")
    print("AUGMENTATION SELECIONADO")
    print("========================================\n")

    print(f"Configuração escolhida: {augmentation_name}")
    
    selected_raw_results = [
        result
        for result in experiment_0_results
        if result["augmentation"] == best_augmentation
    ]

    print("\n========================================")
    print("EXPERIMENTO 1")
    print("========================================\n")
    
    experiment_1_results = selected_raw_results.copy()

    for dataset_type, reference, normalization in EXPERIMENT_1:

        experiment_name = get_experiment_name(
            dataset_type,
            reference,
            normalization
        )

        for seed in SEEDS:

            print(
                f"\nExecutando: {experiment_name} "
                f"| Seed: {seed}"
            )

            experiment_dir = (
                results_dir
                / "experiment_1"
                / experiment_name
                / f"seed_{seed}"
            )

            experiment_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            result = train_experiment(
                dataset_type,
                reference,
                normalization,
                experiment_dir,
                seed,
                best_augmentation
            )

            experiment_1_results.append(result)

            print(
                f"Validation F1: "
                f"{result['validation_metrics']['f1']:.4f}"
            )

    best_normalizations = calculate_best_normalizations(
        experiment_1_results
    )

    print("\n========================================")
    print("MELHORES NORMALIZAÇÕES")
    print("========================================\n")

    for normalization in best_normalizations:
        print(normalization)

    print("\n========================================")
    print("EXPERIMENTO 2")
    print("========================================\n")

    experiment_2 = create_experiment_2(
        best_normalizations
    )

    experiment_2_results = selected_raw_results.copy()

    for dataset_type, reference, normalization in experiment_2:

        experiment_name = get_experiment_name(
            dataset_type,
            reference,
            normalization
        )

        for seed in SEEDS:

            print(
                f"\nExecutando: {experiment_name} "
                f"| Seed: {seed}"
            )

            experiment_dir = (
                results_dir
                / "experiment_2"
                / experiment_name
                / f"seed_{seed}"
            )

            experiment_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            result = train_experiment(
                dataset_type,
                reference,
                normalization,
                experiment_dir,
                seed,
                best_augmentation
            )

            experiment_2_results.append(result)

            print(
                f"Validation F1: "
                f"{result['validation_metrics']['f1']:.4f}"
            )

    print("\n========================================")
    print("AVALIAÇÃO FINAL NO TESTE")
    print("========================================\n")

    all_results = []

    for experiment in experiment_1_results:

        dataset_dir = get_dataset_dir(
            experiment["dataset_type"],
            experiment["reference"],
            experiment["normalization"]
        )

        test_loader = create_evaluation_loader(
            SPLITS_DIR / "test.csv",
            dataset_dir,
            BATCH_SIZE,
            NUM_WORKERS
        )

        metrics = evaluate_test(
            experiment,
            test_loader
        )

        all_results.append({
            "experiment": "experiment_1",
            "dataset_type": experiment["dataset_type"],
            "reference": experiment["reference"],
            "normalization": experiment["normalization"],
            "seed": experiment["seed"],
            **metrics
        })

    for experiment in experiment_2_results:
        
        if experiment["dataset_type"] == "raw":
            continue

        dataset_dir = get_dataset_dir(
            experiment["dataset_type"],
            experiment["reference"],
            experiment["normalization"]
        )

        test_loader = create_evaluation_loader(
            SPLITS_DIR / "test.csv",
            dataset_dir,
            BATCH_SIZE,
            NUM_WORKERS
        )

        metrics = evaluate_test(
            experiment,
            test_loader
        )

        all_results.append({
            "experiment": "experiment_2",
            "dataset_type": experiment["dataset_type"],
            "reference": experiment["reference"],
            "normalization": experiment["normalization"],
            "seed": experiment["seed"],
            **metrics
        })

    # SALVAR RESULTADOS CONSOLIDADOS
    save_all_results(
        all_results,
        results_dir / "classification_results.csv"
    )
    
    save_classification_summary(
        all_results,
        results_dir / "classification_summary.csv"
    )

    print("\n========================================")
    print("RESULTADOS FINAIS")
    print("========================================\n")

    for result in all_results:

        print(
            f"{result['experiment']} | "
            f"{result['normalization']} | "
            f"{result['reference']} | "
            f"Seed: {result['seed']} | "
            f"Accuracy: {result['accuracy']:.4f} | "
            f"F1: {result['f1']:.4f} | "
            f"AUC: {result['auc']:.4f}"
        )


if __name__ == "__main__":
    main()
