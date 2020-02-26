from collections import OrderedDict
import torch
import torch.nn as nn


class Net(nn.Module):
    """
    Input - 1x200x200
    C1 - 6@196x196 (5x5 kernel)
    relu
    S2 - 6@98x98 (2x2 kernel, stride 2) Subsampling
    C3 - 16@94x94 (5x5 kernel)
    relu
    S4 - 16@47x47 (2x2 kernel, stride 2) Subsampling
    C5 - 32@43x43 (5x5 kernel)
    relu
    S5 - 32@21x21 (3x3 kernel, stride 2) Subsampling
    C6 - 64@17x17 (5x5 kernel)
    relu
    S6 - 64@8x8 (3x3 kernel, stride 2) Subsampling
    C7 - 120@1x1 (8x8 kernel)
    relu
    F8 - 84
    relu
    F9 - 10
    relu
    F10 - 3
    """

    def __init__(self):
        super(Net, self).__init__()
        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 6, kernel_size=5)),
            ('relu1', nn.ReLU()),
            ('s2', nn.MaxPool2d(kernel_size=2, stride=2)),
            ('c3', nn.Conv2d(6, 16, kernel_size=5)),
            ('relu3', nn.ReLU()),
            ('s4', nn.MaxPool2d(kernel_size=2, stride=2)),
            ('c5', nn.Conv2d(16, 32, kernel_size=5)),
            ('relu5', nn.ReLU()),
            ('s5', nn.MaxPool2d(kernel_size=3, stride=2)),
            ('c6', nn.Conv2d(32, 64, kernel_size=5)),
            ('relu6', nn.ReLU()),
            ('s6', nn.MaxPool2d(kernel_size=3, stride=2)),
            ('c7', nn.Conv2d(64, 120, kernel_size=8)),
            ('relu7', nn.ReLU()),
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f8', nn.Linear(120, 84)),
            ('relu8', nn.ReLU()),
            ('f9', nn.Linear(84, 10)),
            ('relu9', nn.ReLU()),
            ('f10', nn.Linear(10, 3)),
        ]))

    def forward(self, img: torch.tensor):
        output = self.convnet(img)
        output = output.view(img.size(0), -1)
        output = self.fc(output)
        return output

