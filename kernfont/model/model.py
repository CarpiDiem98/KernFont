import torch
import torch.nn as nn


class AlexNetRegression(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_branch_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.conv_branch_2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.fc_numeric = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
        )

        self.fc_combined = nn.Linear(891136, 1)

    def forward(self, sx, dx):
        sx = self.conv_branch_1(sx)
        sx = sx.view(sx.size(0), -1)
        dx = self.conv_branch_2(dx)
        dx = dx.view(dx.size(0), -1)
        combined = torch.cat((sx, dx), dim=1)
        output = self.fc_combined(combined)
        return output
