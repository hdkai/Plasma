# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from pytest import fixture, mark
from torch import cat, ones, tensor, zeros, zeros_like
from torch.nn import L1Loss, Parameter
from torch.optim import Adam
from .common import tensorread, tensorwrite

from plasma.filters import color_balance
from plasma.losses import ColorOpponentLoss

def test_chroma_optimization ():
    input = tensorread("test/media/conversion/linear.jpg")
    target = tensorread("test/media/conversion/linear_wb.jpg")
    white_balance = Parameter(zeros(1, 2))
    l1_loss = L1Loss()
    chroma_loss = ColorOpponentLoss()
    optimizer = Adam([white_balance], lr=5e-3)
    for _ in range(200):
        # Apply wb
        temp, tint = white_balance.split(1, dim=1)
        weight = cat([temp, tint], dim=1)
        prediction = color_balance(input, weight)
        # Backward
        optimizer.zero_grad()
        #loss = l1_loss(prediction, target)
        loss = chroma_loss(prediction, target)
        loss.backward()
        optimizer.step()
        print(white_balance.data, loss)
    tensorwrite("prediction.jpg", prediction)