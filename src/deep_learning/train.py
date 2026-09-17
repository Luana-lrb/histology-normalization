import torch
from pathlib import Path


def train_one_epoch(
    model,
    dataloader, # fornece os batches de imagens e labels
    criterion, # calcula a loss
    optimizer, # atualiza os pesos do modelo
    device
):

    model.train()

    running_loss = 0.0 # vamos acumular a loss de todos os batches
    correct = 0 # previsões corretas durante a época
    total = 0 # quantidade de imagens processadas

    for images, labels in dataloader:

        # mandando pro hardware
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad() # limpa os gradientes anteriores em cada batch

        outputs = model(images) # retorna [32,4], sendo 4 os logits para cada classe

        loss = criterion(outputs, labels) # quanto a previsão está distante da classe correta

        loss.backward() # backpropagation: calcula os gradientes da loss em relação aos pesos do modelo

        optimizer.step() # atualiza os pesos

        running_loss += loss.item() * images.size(0) # acumula a loss pelo tamanho do batch

        _, predictions = torch.max(outputs, dim=1) # retorna o valor máximo e o índice do valor máximo (classe prevista) para cada imagem

        correct += (predictions == labels).sum().item() # conta os true

        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy

def validate(
    model,
    dataloader,
    criterion,
    device
):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad(): # Não precisamos calcular os gradientes durante a validação

        for images, labels in dataloader: # um batch de cada vez

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, predictions = torch.max(outputs, dim=1)

            correct += (predictions == labels).sum().item()

            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy

def train_model(
    model,
    train_loader,
    validation_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    epochs,
    patience,
    save_path
):
    """
    Treina o modelo e salva o melhor checkpoint
    com base na menor Validation Loss.
    """

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_loss = float("inf") # começa com a pior loss possível (infinito)
    epochs_without_improvement = 0 # contador para o early stopping

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):

        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )

        val_loss, val_acc = validate(
            model=model,
            dataloader=validation_loader,
            criterion=criterion,
            device=device
        )

        scheduler.step(val_loss) # pega a val loss e observa se precisa reduzir a learning rate

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            epochs_without_improvement = 0

            torch.save(
                model.state_dict(),
                save_path
            )
        else:
            epochs_without_improvement += 1

            print(
                f"Sem melhora por "
                f"{epochs_without_improvement}/{patience} épocas."
            )

            if epochs_without_improvement >= patience:
                print("Early stopping acionado.")
                break

        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

    return history
