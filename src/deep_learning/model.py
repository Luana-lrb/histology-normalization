# from torchvision.models import (
#     mobilenet_v2,
#     MobileNet_V2_Weights
# )
# from torch import nn

# from config import (
#     NUM_CLASSES,
#     PRETRAINED,
#     FREEZE_LAYERS,
#     TRAINING_MODE,
#     DROPOUT_RATE
# )


# def create_model():
#     """
#     Cria a MobileNetV2 adaptada para classificação
#     das imagens histológicas.
#     """

#     if PRETRAINED:
#         weights = MobileNet_V2_Weights.DEFAULT
#     else:
#         weights = None

#     model = mobilenet_v2(weights=weights)

#     in_features = model.classifier[1].in_features

#     model.classifier[1] = nn.Linear(
#         in_features,
#         NUM_CLASSES
#     )

#     # Configura o Dropout do classificador
#     model.classifier[0] = nn.Dropout(p=DROPOUT_RATE)

#     if TRAINING_MODE == "partial":

#         # Congela os primeiros módulos do extrator
#         for layer in model.features[:FREEZE_LAYERS]:
#             for parameter in layer.parameters():
#                 parameter.requires_grad = False

#     elif TRAINING_MODE == "classifier_only":

#         # Congela todos os parâmetros do extrator de características
#         for parameter in model.features.parameters():
#             parameter.requires_grad = False

#         # Mantém apenas o classificador treinável
#         for parameter in model.classifier.parameters():
#             parameter.requires_grad = True

#     else:
#         raise ValueError(
#             f"Modo de treinamento não reconhecido: {TRAINING_MODE}"
#         )

#     # Como nenhuma camada é congelada temos um fine-tuning completo

#     return model


from torchvision.models import (
    mobilenet_v2,
    MobileNet_V2_Weights,
    efficientnet_b0,
    EfficientNet_B0_Weights
)

from torch import nn

from config import (
    MODEL_ARCH,
    NUM_CLASSES,
    PRETRAINED,
    FREEZE_LAYERS,
    TRAINING_MODE,
    DROPOUT_RATE
)


def create_model():
    """
    Cria a arquitetura selecionada e configura o
    fine-tuning para classificação histológica.
    """

    # 1. Selecionar a arquitetura
    if MODEL_ARCH == "mobilenet_v2":

        weights = (
            MobileNet_V2_Weights.DEFAULT
            if PRETRAINED else None
        )

        model = mobilenet_v2(weights=weights)

    elif MODEL_ARCH == "efficientnet_b0":

        weights = (
            EfficientNet_B0_Weights.DEFAULT
            if PRETRAINED else None
        )

        model = efficientnet_b0(weights=weights)

    else:
        raise ValueError(
            f"Arquitetura não suportada: {MODEL_ARCH}"
        )

    # 2. Adaptar o classificador
    in_features = model.classifier[1].in_features

    model.classifier[0] = nn.Dropout(
        p=DROPOUT_RATE
    )

    model.classifier[1] = nn.Linear(
        in_features,
        NUM_CLASSES
    )

    # 3. Configurar o congelamento
    if TRAINING_MODE == "partial":

        if not 0 <= FREEZE_LAYERS < len(model.features):
            raise ValueError(
                f"FREEZE_LAYERS={FREEZE_LAYERS} inválido "
                f"para {MODEL_ARCH}, que possui "
                f"{len(model.features)} módulos."
            )

        for layer in model.features[:FREEZE_LAYERS]:
            for parameter in layer.parameters():
                parameter.requires_grad = False

    elif TRAINING_MODE == "classifier_only":

        for parameter in model.features.parameters():
            parameter.requires_grad = False

        for parameter in model.classifier.parameters():
            parameter.requires_grad = True

    else:
        raise ValueError(
            f"Modo de treinamento inválido: {TRAINING_MODE}"
        )

    return model
