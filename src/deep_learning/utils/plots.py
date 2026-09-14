import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import numpy as np

def plot_confusion_matrix(confusion, class_names, save_path):
    fig, ax = plt.subplots(figsize=(7, 6))

    image = ax.imshow(
        confusion,
        cmap="Blues",
        vmin=0,
        vmax=confusion.max()
    )

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))

    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe verdadeira")
    ax.set_title("Matriz de Confusão")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            value = confusion[i, j]

            text_color = "white" if value > confusion.max() / 2 else "black"

            ax.text(
                j,
                i,
                value,
                ha="center",
                va="center",
                color=text_color
            )

    fig.colorbar(
        image,
        ax=ax,
        label="Quantidade de amostras"
    )

    fig.tight_layout()

    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)
    
def plot_training_history(history, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(
        epochs,
        history["train_loss"],
        label="Treino"
    )

    ax.plot(
        epochs,
        history["val_loss"],
        label="Validação"
    )

    ax.set_xlabel("Época")
    ax.set_ylabel("Loss")
    ax.set_title("Curva de Loss")
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)
    
def plot_training_accuracy(history, save_path):
    epochs = range(1, len(history["train_acc"]) + 1)

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(
        epochs,
        history["train_acc"],
        label="Treino"
    )

    ax.plot(
        epochs,
        history["val_acc"],
        label="Validação"
    )

    ax.set_xlabel("Época")
    ax.set_ylabel("Accuracy")
    ax.set_title("Curva de Accuracy")
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)
    
def plot_roc_curve(labels, probabilities, class_names, save_path):
    probabilities = np.array(probabilities)
    labels_binarized = label_binarize(
        labels,
        classes=range(len(class_names))
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(
            labels_binarized[:, i],
            probabilities[:, i]
        )

        roc_auc = auc(fpr, tpr)

        ax.plot(
            fpr,
            tpr,
            label=f"{class_name} (AUC = {roc_auc:.2f})"
        )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Aleatório"
    )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Curva ROC")
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)