import torch.nn as nn

class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=3, num_filters=64, num_layers=3):
        super(PatchGANDiscriminator, self).__init__()

        layers = []
        layers.append(nn.Conv2d(in_channels, num_filters, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Add additional layers
        for i in range(num_layers - 1):
            layers.append(nn.Conv2d(num_filters * (2 ** i), num_filters * (2 ** (i+1)), kernel_size=4, stride=2, padding=1))
            layers.append(nn.InstanceNorm2d(num_filters * (2 ** (i+1))))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Add output layer
        layers.append(nn.Conv2d(num_filters * (2 ** (num_layers - 1)), 1, kernel_size=4, stride=1, padding=1))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x