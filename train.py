import itertools
import ipdb
import numpy as np
import os
import visdom
import torch
import torch.nn as nn
from model import Net
from torch.utils.data import DataLoader
import torch.optim as optim
from typing import Tuple
from utils import iou, noisy_circle, normalize


class DIOULOSS(nn.Module):
    def __init__(self, mn: float, std: float, w: int, h: int):
        super(DIOULOSS, self).__init__()
    def forward(self, x, y):
        # import ipdb
        # ipdb.set_trace()
        mse = torch.mean(torch.sum((x - y)**2, axis=1))
        return mse


def generate_training_data(n: int, train_perc: float=0.8) -> Tuple[np.ndarray, np.ndarray, float, float]:
    np.random.seed(0)
    def _list_of_tuples(list1, list2):
        return list(map(lambda x, y: (x, y), list1, list2))
    mean_params = None
    params_list = []
    image_list = []
    for i in range(n):
        params, img = noisy_circle(200, 50, 2)
        params_list.append(params)
        image_list.append(np.expand_dims(img, axis=0))

    # Split data to train and val
    train_params_list, val_params_list = np.array(params_list[0:int(n * train_perc)]), np.array(params_list[int(n * train_perc):])
    train_image_list, val_image_list = np.array(image_list[0:int(n * train_perc)]), np.array(image_list[int(n * train_perc):])

    # Normalize output to make optimization simpler
    mn_params, std_params = np.mean(train_params_list, axis=0), np.std(train_params_list, axis=0)
    train_params_list = normalize(train_params_list, mn_params, std_params)
    val_params_list = normalize(val_params_list, mn_params, std_params)

    # Output train and test data
    train_params_list = list(map(tuple, train_params_list))
    val_params_list  = list(map(tuple, val_params_list))
    train_data = _list_of_tuples(train_params_list, train_image_list)
    val_data = _list_of_tuples(val_params_list, val_image_list)

    return train_data, val_data, mn_params, std_params


# Generate data
train_data, val_data, mn, std = generate_training_data(n=10)
data_train_loader = DataLoader(train_data, batch_size=256, shuffle=True, num_workers=8)
data_test_loader = DataLoader(val_data, batch_size=1024, num_workers=8)

net = Net().float()
viz = visdom.Visdom()
criterion = DIOULOSS()# nn.MSELoss()
optimizer =optim.Adadelta(net.parameters())

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
    loss_list, batch_list, idx = [], [], 0
    f = open("logging.txt", "a+")
    for i, (labels, images) in enumerate(data_train_loader):
        optimizer.zero_grad()
        # import ipdb
        # ipdb.set_trace()
        labels = torch.stack(labels).T.float()
        output = net(images.float())

        loss = criterion(output, labels)

        loss_list.append(loss.detach().cpu().item())
        batch_list.append(i + 1)

        if i % 10 == 0:
            print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.detach().cpu().item()))
            print('Prediction: {pred} Label: {lab}'.format(pred=output[0:5], lab=labels[0:5]))
        if epoch % 50 == 0 and i == 0:
            torch.save(net.state_dict(), f"checkpoints/model{epoch}")
            f.write(f"{epoch}, {i}, {loss.detach().cpu().item()} \n")

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
    for i, (labels, images) in enumerate(data_test_loader):
        labels = torch.stack(labels).T.float()
        output = net(images.float())
        avg_loss += criterion(output, labels)
        pred = output.detach()
        print('Prediction: {pred} Label: {lab}'.format(pred=pred[0:10], lab=labels[0:10]))
        # total_correct += pred.eq(labels.view_as(pred)).sum()
    avg_loss /= len(val_data)
    print('Test Avg. Loss: %f' % (avg_loss.detach().cpu().item()))


def train_and_test(epoch):
    train(epoch)
    test()


def main():
    if os.path.exists("logging.txt"):
        os.remove("logging.txt")
        f = open("logging.txt", "w+")
        f.close()
    for e in range(1, 1000):
        train_and_test(e)


if __name__ == '__main__':
    main()