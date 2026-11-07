"""Clases originales de NB04a, derivadas de NB02 y NB03."""
import torch
from torch import nn

class BloqueConv(nn.Module):
    """Dos convoluciones 3x3 + ReLU (idéntico a NB02/NB03)."""

    def __init__(self, canales_entrada, canales_salida):
        super().__init__()
        self.capas = nn.Sequential(nn.Conv2d(canales_entrada, canales_salida, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(canales_salida, canales_salida, kernel_size=3, padding=1), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.capas(x)

class EncoderDecoder(nn.Module):
    """A0: Encoder-Decoder SIN conexiones skip (NB02)."""

    def __init__(self):
        super().__init__()
        self.encoder1 = BloqueConv(3, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = BloqueConv(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = BloqueConv(32, 64)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.decoder1 = BloqueConv(64, 32)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.decoder2 = BloqueConv(32, 16)
        self.salida = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        x = self.encoder1(x)
        x = self.pool1(x)
        x = self.encoder2(x)
        x = self.pool2(x)
        x = self.bottleneck(x)
        x = self.up1(x)
        x = self.decoder1(x)
        x = self.up2(x)
        x = self.decoder2(x)
        return self.salida(x)

class MiniUNet(nn.Module):
    """A1: misma red que A0 + 2 conexiones skip por concatenación (NB03)."""

    def __init__(self):
        super().__init__()
        self.encoder1 = BloqueConv(3, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = BloqueConv(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = BloqueConv(32, 64)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.decoder1 = BloqueConv(64 + 32, 32)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.decoder2 = BloqueConv(32 + 16, 16)
        self.salida = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        skip1 = self.encoder1(x)
        x = self.pool1(skip1)
        skip2 = self.encoder2(x)
        x = self.pool2(skip2)
        x = self.bottleneck(x)
        x = self.up1(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.decoder1(x)
        x = self.up2(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.decoder2(x)
        return self.salida(x)

class BCEDiceLoss(nn.Module):
    """BCE + Dice (1:1), idéntica a NB02/NB03."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, logits, mascaras):
        loss_bce = self.bce(logits, mascaras)
        prob = torch.sigmoid(logits)
        dims = (1, 2, 3)
        inter = torch.sum(prob * mascaras, dim=dims)
        dice = (2 * inter + self.smooth) / (torch.sum(prob, dim=dims) + torch.sum(mascaras, dim=dims) + self.smooth)
        return loss_bce + (1 - dice.mean())
