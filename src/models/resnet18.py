import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def build_resnet18(
    pretrained: bool = True,
    strategy: str = "finetune",
):
    """Build ResNet18 for binary histopathology classification.

    Strategies:
      - frozen: train only the final classifier
      - finetune: train layer4 + final classifier
      - full: train the complete network
    """
    if strategy not in {"frozen", "finetune", "full"}:
        raise ValueError(
            "strategy must be one of: frozen, finetune, full"
        )

    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)

    # Freeze everything first.
    for parameter in model.parameters():
        parameter.requires_grad = False

    if strategy == "full":
        for parameter in model.parameters():
            parameter.requires_grad = True

    elif strategy == "finetune":
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True

    # The classifier is always trainable.
    model.fc = nn.Linear(model.fc.in_features, 2)

    return model


def trainable_parameters(model):
    return [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]


def count_trainable_parameters(model) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
