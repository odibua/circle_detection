from collections import OrderedDict
import numpy as np
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

class DIOULOSS(nn.Module):
    """
    Calculates a loss based on the location of the circles center and it's radius, along with the IOU between
    the circles formed by these predicted quantities and labels
    """
    def __init__(self, mn: np.ndarray, std: np.ndarray, w: int, h: int, mn_label: np.ndarray = None, std_label: np.ndarray = None):
        super(DIOULOSS, self).__init__()
        self.mn, self.std = mn, std
        self.mn_label, self.std_label = mn_label, std_label
        self.w, self.h = w, h

    def forward(self, pred, label):
        n = pred.size()[0]
        # import ipdb
        # ipdb.set_trace()

        # Initializes tensors that are used to calculate the IOU between labels and predictions
        self.x_pred, self.y_pred = torch.zeros([n, self.w], dtype=torch.int32), torch.zeros([n, self.h], dtype=torch.int32)
        self.x_label, self.y_label = torch.zeros([n, self.w], dtype=torch.int32), torch.zeros([n, self.w], dtype=torch.int32)

        # Dimensionalize the predictions and labels
        pred_dim = self.denormalize_gaussian(pred, 'pred')
        label_dim = self.denormalize_gaussian(label, 'label')

        # Get range of x and y values for predictions and labels
        x_pred_ranges, y_pred_ranges = self.get_range(n, pred_dim)
        x_label_ranges, y_label_ranges = self.get_range(n, label_dim)

        # Calculate IOU
        iou = torch.tensor(0)
        for idx in range(n):
            self.x_pred[idx, x_pred_ranges[idx][0]:x_pred_ranges[idx][1]], self.x_label[idx, x_label_ranges[idx][0]:x_label_ranges[idx][1]] = 1, 1
            self.y_pred[idx, y_pred_ranges[idx][0]:y_pred_ranges[idx][1]], self.y_label[idx, y_label_ranges[idx][0]:y_label_ranges[idx][1]] = 1, 1
            iou = iou + torch.sum(self.x_pred & self.x_label, dtype=torch.float) * torch.sum(self.y_pred & self.y_label, dtype=torch.float)\
                  / (torch.sum(self.x_pred | self.x_label, dtype=torch.float) * torch.sum(self.y_pred | self.y_label, dtype=torch.float))
        iou = iou/torch.tensor(n)

        diou = torch.mean(torch.sum((pred - label)**2, axis=1)) + (1 - iou)
        return diou, iou

    def get_range(self, n: int, params: np.ndarray):
        x_low = torch.max(torch.cat(((params[:, 1] - params[:, 2]).reshape(-1, 1), torch.zeros([n, 1], dtype=torch.double)), dim=1), axis=1).values
        x_up = torch.min(torch.cat(((params[:, 1] + params[:, 2]).reshape(-1, 1), torch.ones([n, 1], dtype=torch.double)*torch.tensor(self.w)), dim=1), axis=1).values
        y_low = torch.max(torch.cat(((params[:, 0] - params[:, 2]).reshape(-1, 1), torch.zeros([n, 1], dtype=torch.double)), dim=1), axis=1).values
        y_up = torch.min(torch.cat(((params[:, 0] + params[:, 2]).reshape(-1, 1), torch.ones([n, 1], dtype=torch.double)*torch.tensor(self.h)), dim=1), axis=1).values

        return torch.tensor(torch.cat((x_low.reshape(-1, 1), x_up.reshape(-1, 1)), dim=1), dtype=torch.long), torch.tensor(torch.cat((y_low.reshape(-1, 1), y_up.reshape(-1, 1)), dim=1), dtype=torch.long)

    def denormalize_gaussian(self, params, type):
        if type == 'pred':
            return params * torch.tensor(self.std) + torch.tensor(self.mn)
        elif type == 'label':
            if not isinstance(self.mn_label, np.ndarray):
                return params * torch.tensor(self.std) + torch.tensor(self.mn)
            else:
                return params * torch.tensor(self.std_label) + torch.tensor(self.mn_label)


