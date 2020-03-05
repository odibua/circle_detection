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
from utils import (
    create_log_files,
    iou,
    noisy_circle,
    normalize)


class DIOULOSS(nn.Module):
    def __init__(self, mn: np.ndarray, std: np.ndarray, w: int, h: int):
        super(DIOULOSS, self).__init__()
        self.mn, self.std = mn, std
        self.w, self.h = w, h
        self.img_label = None
        self.img_pred = None
    def forward(self, pred, label):
        n = pred.size()[0]
        # import ipdb
        # ipdb.set_trace()
        if self.img_label == None:
            self.x_pred, self.y_pred = torch.zeros([n, self.w], dtype=torch.int32), torch.zeros([n, self.h], dtype=torch.int32)
            self.x_label, self.y_label = torch.zeros([n, self.w], dtype=torch.int32), torch.zeros([n, self.w], dtype=torch.int32)
        else:
            self.x_pred, self.y_pred = self.x_pred*torch.tensor(0), self.y_pred*torch.tensor(0)
            self.x_label, self.y_label = self.x_label*torch.tensor(0), self.y_label*torch.tensor(0)

        pred_dim = pred*torch.tensor(self.std) + torch.tensor(mn)
        label_dim = label*torch.tensor(self.std) + torch.tensor(mn)
        x_pred_ranges, y_pred_ranges = self.get_range(n, pred_dim)
        x_label_ranges, y_label_ranges = self.get_range(n, label_dim)

        # self.img_pred = self.fill_img(self.img_pred, n, x_pred_ranges, y_pred_ranges)
        # self.img_label = self.fill_img(self.img_label, n, x_label_ranges, y_label_ranges)

        iou = torch.tensor(0)
        for idx in range(n):
            self.x_pred[idx, x_pred_ranges[idx][0]:x_pred_ranges[idx][1]], self.x_label[idx, x_label_ranges[idx][0]:x_label_ranges[idx][1]] = 1, 1
            self.y_pred[idx, y_pred_ranges[idx][0]:y_pred_ranges[idx][1]], self.y_label[idx, y_label_ranges[idx][0]:y_label_ranges[idx][1]] = 1, 1
            iou = iou + torch.sum(self.x_pred & self.x_label, dtype=torch.float) * torch.sum(self.y_pred & self.y_label, dtype=torch.float)\
                  / (torch.sum(self.x_pred | self.x_label, dtype=torch.float) * torch.sum(self.y_pred | self.y_label, dtype=torch.float))
        iou = iou/torch.tensor(n)
        # print(f"IOU {iou}")
        # import ipdb
        # ipdb.set_trace()
        mse = torch.mean(torch.sum((pred - label)**2, axis=1)) + (1 - iou)
        return mse, iou

    def get_range(self, n, params):
        x_low = torch.max(torch.cat(((params[:, 1] - params[:, 2]).reshape(-1, 1), torch.zeros([n, 1], dtype=torch.double)), dim=1), axis=1).values
        x_up = torch.min(torch.cat(((params[:, 1] + params[:, 2]).reshape(-1, 1), torch.ones([n, 1], dtype=torch.double)*torch.tensor(self.w)), dim=1), axis=1).values
        y_low = torch.max(torch.cat(((params[:, 0] - params[:, 2]).reshape(-1, 1), torch.zeros([n, 1], dtype=torch.double)), dim=1), axis=1).values
        y_up = torch.min(torch.cat(((params[:, 0] + params[:, 2]).reshape(-1, 1), torch.ones([n, 1], dtype=torch.double)*torch.tensor(self.h)), dim=1), axis=1).values

        return torch.tensor(torch.cat((x_low.reshape(-1, 1), x_up.reshape(-1, 1)), dim=1), dtype=torch.long), torch.tensor(torch.cat((y_low.reshape(-1, 1), y_up.reshape(-1, 1)), dim=1), dtype=torch.long)

    @staticmethod
    def fill_img(img, n, x_ranges, y_ranges):
        # import ipdb
        # ipdb.set_trace()
        for idx in range(n):
            img[idx, y_ranges[idx, 0]:y_ranges[idx, 1], x_ranges[idx, 0]:x_ranges[idx, 1]] = 1
        return img


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
train_data, val_data, mn, std = generate_training_data(n=500)
data_train_loader = DataLoader(train_data, batch_size=256, shuffle=True, num_workers=8)
data_test_loader = DataLoader(val_data, batch_size=1284, shuffle=True, num_workers=8)

net = Net().float()
viz = visdom.Visdom()
criterion = DIOULOSS(mn=mn, std=std, w=200, h=200)# nn.MSELoss()
optimizer =optim.Adadelta(net.parameters())

cur_batch_win = None
cur_batch_win_opts = {
    'title': 'Epoch Loss Trace',
    'xlabel': 'Batch Number',
    'ylabel': 'Loss',
    'width':  200,
    'height': 200,
}


def train(epoch: int):
    global cur_batch_win
    net.train()
    loss_list, batch_list, idx = [], [], 0
    f = open("train_logs.txt", "a+")
    for i, (labels, images) in enumerate(data_train_loader):
        optimizer.zero_grad()
        # import ipdb
        # ipdb.set_trace()
        labels = torch.stack(labels).T.float()
        output = net(images.float())

        loss, iou = criterion(output, labels)

        loss_list.append(loss.detach().cpu().item())
        batch_list.append(i + 1)

        if i % 10 == 0:
            print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.detach().cpu().item()))
            # print('Prediction: {pred} Label: {lab}'.format(pred=output[0:5], lab=labels[0:5]))
        if epoch % 100 == 0 and i == 0:
            torch.save(
                {'state_dict': net.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'epoch': epoch + 1,
                 },
                f"checkpoints/model{epoch}")
            f.write(f"{epoch}, {i}, {loss.detach().cpu().item()}, {iou} \n")

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
    avg_loss, avg_iou = 0.0, 0.0
    for i, (labels, images) in enumerate(data_test_loader):
        labels = torch.stack(labels).T.float()
        output = net(images.float())
        loss, iou = criterion(output, labels)
        avg_loss += loss
        avg_iou += iou
        # print('Prediction: {pred} Label: {lab}'.format(pred=pred[0:10], lab=labels[0:10]))
        # total_correct += pred.eq(labels.view_as(pred)).sum()
    avg_loss /= (i+1)
    avg_iou /= (i+1)
    # print('Test Avg. Loss: %f' % (avg_loss.detach().cpu().item()))
    return avg_loss, avg_iou


def train_and_test(epoch):
    f = open("test_logs.txt", "a+")
    train(epoch)
    avg_loss, avg_iou = test()
    f.write(f"{epoch}, {avg_loss} , {avg_iou}\n")


def main():
    create_log_files("train_logs.txt", 'epoch, batch, loss, iou\n')
    create_log_files("test_logs.txt", 'epoch, avg_loss, avg_iou\n')
    for e in range(1, 3000):
        train_and_test(e)


if __name__ == '__main__':
    main()