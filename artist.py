import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

class BasicEncoder(nn.Module):
    def __init__(self, info_channels, layers=10):
        super(BasicEncoder, self).__init__()


    def forward(self, image, mask):
        info =  torch.cat((image, mask), dim=1)
        info = self.enc(info)
        return nn.Tanh(self.dec(info))

    @staticmethod
    def _block(in_channels, features, name):

        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )

class CustomSequential(nn.Module):
    def __init__(self, *args):
        super(CustomSequential, self).__init__()
        self.layers = nn.Sequential(*args)

    def forward(self, image, mask):
        torch.cat((image, mask), dim = 1)
        for layer in self.layers:
            if isinstance(layer, nn.Conv2d):
                # Пример использования дополнительного параметра
                x = layer(x)
                # Здесь вы можете использовать additional_param по вашему усмотрению
            else:
                x = layer(x)
        return x

