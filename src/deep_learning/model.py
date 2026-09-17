from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights
)
from torch import nn # é o módulo de redes neurais do PyTorch

from config import NUM_CLASSES, PRETRAINED


def create_model():
    """
    Cria a EfficientNet-B0 adaptada para classificação
    das imagens histológicas.
    """

    if PRETRAINED:
        weights = EfficientNet_B0_Weights.DEFAULT # usando os pesos padrões pré-treinados no ImageNet
    else:
        weights = None

    model = efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features # obtendo o número de features de entrada da camada final

    model.classifier[1] = nn.Linear(
        in_features, # o número de entradas permanece o mesmo
        NUM_CLASSES # o número de saídas é o número de classes do nosso dataset
    ) # retorna logits
    
    # Como nenhuma camada é congelada temos um fine-tuning completo

    return model
