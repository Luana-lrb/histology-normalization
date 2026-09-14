import torch
from torch.utils.data import DataLoader

from dataset import HistologyDataset
from metrics import calculate_all_metrics, calculate_confusion_matrix
from model import create_model
from transforms import test_transform

def load_model(model_path, device):
    model = create_model()
    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )
    model = model.to(device)
    model.eval()

    return model

def create_evaluation_loader(csv_file, root_dir, batch_size, num_workers):
    dataset = HistologyDataset(
        csv_file=csv_file,
        root_dir=root_dir,
        transform=test_transform
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return loader

def predict(model, dataloader, device):
    all_labels = []
    all_predictions = []
    all_probabilities = []

    model.eval()

    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(device)

            outputs = model(images)

            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

            all_labels.extend(labels.numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    return all_labels, all_predictions, all_probabilities

def evaluate_model(labels, predictions, probabilities):
    metrics = calculate_all_metrics(
        labels,
        predictions,
        probabilities
    )

    confusion = calculate_confusion_matrix(
        labels,
        predictions
    )

    return metrics, confusion