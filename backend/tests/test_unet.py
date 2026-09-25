import torch

from src.models.unet import UNet


def test_unet_preserves_patch_shape():
    model = UNet(in_channels=3, out_channels=1, base_channels=8)
    x = torch.randn(2, 3, 128, 128)
    y = model(x)
    assert tuple(y.shape) == (2, 1, 128, 128)
