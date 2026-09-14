"""
Modelo utilizado na classificação das imagens histológicas.
"""

from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights
)
from torch import nn

from config import NUM_CLASSES, PRETRAINED


def create_model():
    """
    Cria a EfficientNet-B0 adaptada para classificação
    das imagens histológicas.
    """

    if PRETRAINED:
        weights = EfficientNet_B0_Weights.DEFAULT
    else:
        weights = None

    model = efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        in_features,
        NUM_CLASSES
    )

    return model