import itertools
import ipdb
import numpy as np
import visdom
import torch
import torch.nn as nn
from collections import OrderedDict
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from utils import iou, noisy_circle
from shapely.geometry import Point, GeometryCollection


def giou(params0, params1):
    row0, col0, rad0 = params0
    row1, col1, rad1 = params1

    shape0 = Point(row0, col0).buffer(rad0)
    shape1 = Point(row1, col1).buffer(rad1)
    collection = GeometryCollection([shape0, shape1])

    return (
            iou(params0, params1) - collection.difference(shape0.union(shape1)).area /
            collection.area
    )

class GIOULOSS(nn.Module):

    def __init__(self):
        super(GIOULOSS, self).__init()

    def forward(self, x, y):
        x_npy, y_npy = x.data.numpy(), y.data.numpy()
        mse = np.sum(np.sum((x_npy - y_npy)**2, axis=0))

class LeNet5(nn.Module):
    """
    Input - 1x200x200
    C1 - 6@196x196 (5x5 kernel)
    tanh
    S2 - 6@98x98 (2x2 kernel, stride 2) Subsampling
    C3 - 16@94x94 (5x5 kernel)
    tanh
    S4 - 16@47x47 (2x2 kernel, stride 2) Subsampling
    C5 - 32@43x43 (5x5 kernel)
    tanh
    S5 - 32@21x21 (3x3 kernel, stride 2) Subsampling
    C6 - 64@17x17 (5x5 kernel)
    tanh
    S6 - 64@8x8 (3x3 kernel, stride 2) Subsampling
    C7 - 120@1x1 (8x8 kernel)
    tanh

    F6 - 84
    tanh
    F7 - 10 (Output)
    """

    def __init__(self):
        super(LeNet5, self).__init__()
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
            ('f6', nn.Linear(120, 84)),
            ('relu6', nn.ReLU()),
            ('f7', nn.Linear(84, 10)),
            ('f8', nn.Linear(10, 3))
        ]))

    def forward(self, img):
        output = self.convnet(img)
        output = output.view(img.size(0), -1)
        output = self.fc(output)
        return output


def generate_training_data(n, train_perc=0.8):
    np.random.seed(0)
    param_image_list = []

    for i in range(n):
        params, img = noisy_circle(200, 50, 2)
        param_image_list.append((params, np.expand_dims(img, axis=0)))
    train_data, val_data = param_image_list[0:int(n * train_perc)], param_image_list[int(n * train_perc):]

    return train_data, val_data


# Generate data
train_data, val_data = generate_training_data(n=2000)

# class CircleDataset(Dataset):
#     def __init__(self, data):
#         self.samples = data

data_train_loader = DataLoader(train_data, batch_size=256, shuffle=False, num_workers=8)
data_test_loader = DataLoader(val_data, batch_size=1024, num_workers=8)

net = LeNet5().float()
viz = visdom.Visdom()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=2e-3)

cur_batch_win = None
cur_batch_win_opts = {
    'title': 'Epoch Loss Trace',
    'xlabel': 'Batch Number',
    'ylabel': 'Loss',
    'width':  200,
    'height': 200,
}


def train(epoch):
    global cur_batch_win
    net.train()
    loss_list, batch_list = [], []
    for i, (labels, images) in enumerate(data_train_loader):
        optimizer.zero_grad()
        # import ipdb
        # ipdb.set_trace()
        output = net(images.float())

        loss = criterion(output, labels)

        loss_list.append(loss.detach().cpu().item())
        batch_list.append(i + 1)

        if i % 10 == 0:
            print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.detach().cpu().item()))

        # Update Visualization
        if viz.check_connection():
            cur_batch_win = viz.line(torch.Tensor(loss_list), torch.Tensor(batch_list),
                                     win=cur_batch_win, name='current_batch_loss',
                                     update=(None if cur_batch_win is None else 'replace'),
                                     opts=cur_batch_win_opts)

        loss.backward()
        optimizer.step()


def test():
    net.eval()
    total_correct = 0
    avg_loss = 0.0
    for i, (images, labels) in enumerate(data_test_loader):
        output = net(images)
        avg_loss += criterion(output, labels).sum()
        pred = output.detach().max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()

    avg_loss /= len(val_data)
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.detach().cpu().item(), float(total_correct) / len(data_test)))


def train_and_test(epoch):
    train(epoch)
    test()


def main():
    for e in range(1, 2):
        train_and_test(e)


if __name__ == '__main__':
    main()