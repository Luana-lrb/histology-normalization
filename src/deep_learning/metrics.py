from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

def calculate_accuracy(labels, predictions):
    return accuracy_score(labels, predictions)

def calculate_classification_metrics(labels, predictions):
    precision = precision_score(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        labels,
        predictions,
        average="macro", # caulcula pras 4 classes e depois faz a media
        zero_division=0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    
def calculate_auc(labels, probabilities):
    return roc_auc_score(
        labels,
        probabilities,
        multi_class="ovr", # one-vs-rest, calcula a AUC para cada classe contra todas as outras classes
        average="macro" # depois faz a media das AUCs de cada classe
    )
    
def calculate_confusion_matrix(labels, predictions):
    return confusion_matrix(labels, predictions)
    
def calculate_all_metrics(labels, predictions, probabilities):
    metrics = {
        "accuracy": calculate_accuracy(labels, predictions),
        **calculate_classification_metrics(labels, predictions), # os ateriscos desempacotam o dicionario
        "auc": calculate_auc(labels, probabilities)
    }

    return metrics

def save_classification_metrics(metrics, save_path):
    with open(save_path, "w") as file:
        for name, value in metrics.items():
            file.write(f"{name},{value:.4f}\n")
