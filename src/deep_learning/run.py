import torch
from torch import nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import csv
import math
import json
from statistics import mean, stdev

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
    MODEL_ARCH,
    EPOCHS,
    SEEDS,
    PATIENCE,
    CLASS_NAMES,
    TOP_K_NORMALIZATIONS,
    OPTIMIZER_NAME,
    SCHEDULER_NAME,
    SCHEDULER_FACTOR,
    SCHEDULER_PATIENCE,
    PRETRAINED,
    FREEZE_LAYERS,
    TRAINING_MODE,
    DROPOUT_RATE,
    USE_DIFFERENTIAL_LR,
    BACKBONE_LR,
    CLASSIFIER_LR,
    FREEZE_BN_STATS,
    IMAGE_SIZE,
    FREEZE_ALL_BN_STATS
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

    criterion = nn.CrossEntropyLoss() # testei com o label_smoothing

    if OPTIMIZER_NAME == "adam":

        optimizer = Adam(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )

    elif OPTIMIZER_NAME == "adamw":

        if USE_DIFFERENTIAL_LR and TRAINING_MODE == "partial":

            backbone_params = [
                p for p in model.features.parameters()
                if p.requires_grad
            ]

            classifier_params = [
                p for p in model.classifier.parameters()
                if p.requires_grad
            ]

            optimizer = AdamW(
                [
                    {
                        "params": backbone_params,
                        "lr": BACKBONE_LR
                    },
                    {
                        "params": classifier_params,
                        "lr": CLASSIFIER_LR
                    }
                ],
                weight_decay=WEIGHT_DECAY
            )

        else:

            optimizer = AdamW(
                (p for p in model.parameters() if p.requires_grad),
                lr=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY
            )

    else:
        raise ValueError(
            f"Otimizador não reconhecido: {OPTIMIZER_NAME}"
        )

    if SCHEDULER_NAME == "plateau":

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=SCHEDULER_FACTOR,
            patience=SCHEDULER_PATIENCE
        )

    else:
        raise ValueError(
            f"Scheduler não reconhecido: {SCHEDULER_NAME}"
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
    # Salvar histórico completo do treinamento
    history_path = experiment_dir / "training_history.csv"

    with open(history_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow([
            "epoch",
            "train_loss",
            "train_acc",
            "val_loss",
            "val_acc",
            "lr",
            "lr_classifier"
        ])

        for i in range(len(history["train_loss"])):
            writer.writerow([
                i + 1,
                history["train_loss"][i],
                history["train_acc"][i],
                history["val_loss"][i],
                history["val_acc"][i],
                history["lr"][i],
                history["lr_classifier"][i]
            ])

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

    results_dir = (
        RESULTS_DIR
        / "deep_learning"
        / "all_references_01"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=False
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

            print(f"Executando: raw | {augmentation_name} | Seed: {seed}")

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

    # ============================================================
    # REGISTRO DA CONFIGURAÇÃO FINAL
    # ============================================================

    configuration = {
        "model_arch": MODEL_ARCH,
        "pretrained": PRETRAINED,
        "training_mode": TRAINING_MODE,
        "freeze_layers": FREEZE_LAYERS,
        "freeze_bn_stats": FREEZE_BN_STATS,
        "freeze_all_bn_stats": FREEZE_ALL_BN_STATS,

        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "early_stopping_patience": PATIENCE,

        "optimizer": OPTIMIZER_NAME,
        "learning_rate": LEARNING_RATE,
        "use_differential_lr": USE_DIFFERENTIAL_LR,
        "backbone_lr": BACKBONE_LR,
        "classifier_lr": CLASSIFIER_LR,
        "weight_decay": WEIGHT_DECAY,
        "dropout_rate": DROPOUT_RATE,

        "scheduler": SCHEDULER_NAME,
        "scheduler_factor": SCHEDULER_FACTOR,
        "scheduler_patience": SCHEDULER_PATIENCE,

        "loss_function": "CrossEntropyLoss",
        "label_smoothing": 0.0,
        "class_weights": None,

        "augmentation": best_augmentation,
        "augmentation_interpolation": "NEAREST",
        "checkpoint_selection": "min_validation_loss",
        "normalization_selection": "mean_validation_f1",
        "seeds": SEEDS,
        "device": str(DEVICE)
    }

    with open(
        results_dir / "configuration_final.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(configuration, file, indent=4)
    
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
    
    configuration["selected_normalizations"] = best_normalizations

    with open(
        results_dir / "configuration_final.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(configuration, file, indent=4)

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

    # ============================================================
    # SALVAR RESULTADOS DE VALIDAÇÃO
    # ============================================================

    validation_entries = []

    # Experimento 0: com e sem augmentation
    for result in experiment_0_results:

        augmentation_name = (
            "with_augmentation"
            if result["augmentation"]
            else "without_augmentation"
        )

        validation_entries.append((
            f"experiment_0_{augmentation_name}",
            result
        ))

    # Experimento 1: RAW selecionado e normalizações com ref1
    for result in experiment_1_results:
        validation_entries.append(("experiment_1", result))

    # Experimento 2: RAW selecionado, ref2, ref3 e multi-target
    for result in experiment_2_results:
        validation_entries.append(("experiment_2", result))

    for result in experiment_1_results:

        if (
            result["dataset_type"] == "processed"
            and result["reference"] == "ref1"
            and result["normalization"] in best_normalizations
        ):
            validation_entries.append(("experiment_2", result))

    validation_rows = []

    for experiment_name, result in validation_entries:

        validation_rows.append({
            "experiment": experiment_name,
            "dataset_type": result["dataset_type"],
            "reference": result["reference"],
            "normalization": result["normalization"],
            "seed": result["seed"],
            **{
                metric: result["validation_metrics"][metric]
                for metric in [
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "auc"
                ]
            }
        })

    save_all_results(
        validation_rows,
        results_dir / "validation_results.csv"
    )
    save_classification_summary(
        validation_rows,
        results_dir / "validation_summary.csv"
    )

    print("\nResultados de validação salvos com sucesso.")

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

        # resultados do Experimento 1
        result_row = {
            "experiment": "experiment_1",
            "dataset_type": experiment["dataset_type"],
            "reference": experiment["reference"],
            "normalization": experiment["normalization"],
            "seed": experiment["seed"],
            **metrics
        }

        all_results.append(result_row)

        if (
            experiment["dataset_type"] == "processed"
            and experiment["reference"] == "ref1"
            and experiment["normalization"] in best_normalizations
        ):
            all_results.append({
                **result_row,
                "experiment": "experiment_2"
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

        if result["dataset_type"] == "raw":

            condition = "RAW"

        else:

            condition = (
                f"{result['normalization']} | "
                f"{result['reference']}"
            )

        print(
            f"{result['experiment']} | "
            f"{condition} | "
            f"Seed: {result['seed']} | "
            f"Accuracy: {result['accuracy']:.4f} | "
            f"F1: {result['f1']:.4f} | "
            f"AUC: {result['auc']:.4f}"
        )

def run_isolated():

    experiment_id = "opt020"

    experiment_name = (
        f"{experiment_id}_{MODEL_ARCH}_"
        + (
            f"f{FREEZE_LAYERS}"
            if TRAINING_MODE == "partial"
            else "classifier"
        )
        + f"_{OPTIMIZER_NAME}"
    )

    experiment_dir = (
        RESULTS_DIR
        / "deep_learning"
        / "optimization"
        / experiment_name
    )

    if experiment_dir.exists():
        raise FileExistsError(
            f"Experimento já existe: {experiment_dir}"
        )

    experiment_dir.mkdir(parents=True, exist_ok=False)

    configuration = {
        "model_arch": MODEL_ARCH,
        "pretrained": PRETRAINED,
        "training_mode": TRAINING_MODE,
        "dropout_rate": DROPOUT_RATE,
        "label_smoothing": 0.05,
        "freeze_layers": (
            FREEZE_LAYERS
            if TRAINING_MODE == "partial"
            else "all_features"),
        "dataset": "raw",
        "augmentation": True,
        "optimizer": OPTIMIZER_NAME,
        "scheduler": SCHEDULER_NAME,
        "scheduler_factor": SCHEDULER_FACTOR,
        "scheduler_patience": SCHEDULER_PATIENCE,
        "learning_rate": LEARNING_RATE,
        "use_differential_lr": USE_DIFFERENTIAL_LR,
        "backbone_lr": BACKBONE_LR,
        "classifier_lr": CLASSIFIER_LR,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "image_size": IMAGE_SIZE,
        "epochs": EPOCHS,
        "early_stopping_patience": PATIENCE,
        "seeds": SEEDS,
        "freeze_bn_stats": FREEZE_BN_STATS,
        "freeze_all_bn_stats": FREEZE_ALL_BN_STATS,
        "device": str(DEVICE)
    }

    with open(
        experiment_dir / "configuration.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(configuration, file, indent=4)

    validation_results = []
    
    for seed in SEEDS:

        seed_dir = experiment_dir / f"seed_{seed}"

        result = train_experiment(
            dataset_type="raw",
            reference=None,
            normalization=None,
            experiment_dir=seed_dir,
            seed=seed,
            use_augmentation=True
        )

        validation_results.append({
            "seed": seed,
            **result["validation_metrics"]
        })

        print(
            f"Seed {seed} | "
            f"Validation F1: "
            f"{result['validation_metrics']['f1']:.4f}"
        )
   
    validation_path = experiment_dir / "validation_results.csv"

    with open(validation_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["seed", "accuracy", "precision", "recall", "f1", "auc"]
        )

        writer.writeheader()
        writer.writerows(validation_results)

    # Resumo das métricas de validação

    metrics_names = ["accuracy", "precision", "recall", "f1", "auc"]

    summary_path = experiment_dir / "validation_summary.csv"

    with open(summary_path, "w", newline="", encoding="utf-8") as file:

        writer = csv.writer(file)
        writer.writerow(["metric", "mean", "std"])

        for metric in metrics_names:

            values = [
                result[metric]
                for result in validation_results
            ]

            metric_mean = mean(values)
            metric_std = stdev(values) if len(values) > 1 else 0.0

            writer.writerow([
                metric,
                metric_mean,
                metric_std
            ])

            print(
                f"{metric.upper()}: "
                f"{metric_mean:.4f} ± {metric_std:.4f}"
            )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["full", "isolated"],
        default="full"
    )

    args = parser.parse_args()

    if args.mode == "isolated":
        run_isolated()
    else:
        main()
