from model import Net
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import utils


class DIOULOSS(nn.Module):
    """
    Calculates a loss based on the location of the circles center and it's radius, along with the IOU between
    the circles formed by these predicted quantities and labels
    """
    def __init__(self, mn: np.ndarray, std: np.ndarray, w: int, h: int):
        super(DIOULOSS, self).__init__()
        self.mn, self.std = mn, std
        self.w, self.h = w, h

    def forward(self, pred, label):
        n = pred.size()[0]
        # import ipdb
        # ipdb.set_trace()

        # Initializes tensors that are used to calculate the IOU between labels and predictions
        self.x_pred, self.y_pred = torch.zeros([n, self.w], dtype=torch.int32), torch.zeros([n, self.h], dtype=torch.int32)
        self.x_label, self.y_label = torch.zeros([n, self.w], dtype=torch.int32), torch.zeros([n, self.w], dtype=torch.int32)

        # Dimensionalize the predictions and labels
        pred_dim = self.denormalize_gaussian(pred)
        label_dim = self.denormalize_gaussian(label)

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

    def denormalize_gaussian(self, params):
        return params * torch.tensor(self.std) + torch.tensor(self.mn)


def train(epoch: int, model: nn.Module, criterion: nn.Module, optimizer: torch.optim, dataloader: torch.utils.data, mn: np.ndarray, std: np.ndarray):
    model.train()
    loss_list, batch_list, idx = [], [], 0
    f = open("train_logs.txt", "a+")
    for i, (labels, images) in enumerate(dataloader):
        optimizer.zero_grad()
        # import ipdb
        # ipdb.set_trace()
        labels = torch.stack(labels).T.float()
        output = model(images.float())

        loss, iou = criterion(output, labels)

        loss_list.append(loss.detach().cpu().item())
        batch_list.append(i + 1)

        if i % 10 == 0:
            print('Train - Epoch %d, Batch: %d, Loss: %f IOU %f' % (epoch, i, loss.detach().cpu().item(), iou.detach().cpu().item()))
            print('Prediction: {pred} Label: {lab}'.format(pred=output[0:2], lab=labels[0:2]))
        if epoch % 5 == 0 and i == 0:
            torch.save(
                {'state_dict': model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'epoch': epoch + 1,
                 'std': std,
                 'mn': mn,
                 },
                f"checkpoints/model_epoch_{epoch}_batch_{i}")
            f.write(f"{epoch}, {i}, {loss.detach().cpu().item()}, {iou} \n")

        loss.backward()
        optimizer.step()


def test(model: nn.Module, criterion: nn.Module, dataloader: torch.utils.data):
    model.eval()
    avg_loss, avg_iou = 0.0, 0.0
    for i, (labels, images) in enumerate(dataloader):
        labels = torch.stack(labels).T.float()
        output = model(images.float())
        loss, iou = criterion(output, labels)
        avg_loss += loss
        avg_iou += iou
        pred = output.detach()
    avg_loss /= (i+1)
    avg_iou /= (i+1)
    print(f"Test Avg. Loss: {avg_loss.detach().cpu().item()} Test Avg. IOU: {avg_loss.detach().cpu().item()}")
    return avg_loss, avg_iou


def train_and_test(epoch: int, model: nn.Module, criterion: nn.Module, optimizer: torch.optim,
                   data_train_loader: torch.utils.data, data_test_loader: torch.utils.data, mn: np.ndarray, std: np.ndarray):

    train(epoch=epoch, model=model, criterion=criterion, optimizer=optimizer, dataloader=data_train_loader,
          mn=mn, std=std)
    # f = open("test_logs.txt", "a+")
    # if epoch % 20 == 0:
    #     avg_loss, avg_iou = test(model=model, criterion=criterion, dataloader=data_test_loader)
    #     f.write(f"{epoch}, {avg_loss} , {avg_iou}\n")


def main():
    load_checkpoint = True
    # utils.create_log_files("train_logs.txt", 'epoch, batch, loss, iou\n')
    # utils.create_log_files("test_logs.txt", 'epoch, avg_loss, avg_iou\n')

    # Initialize model, loss function, and optimizer
    model = Net().float()
    optimizer = optim.Adadelta(model.parameters())

    epoch_start, mn, std = None, None, None
    if load_checkpoint:
        # Load checkpoint, epoch, model, optimizer and relevant params
        checkpoint = torch.load('checkpoints_n_20000/model_epoch_60_batch_0')
        mn, std = checkpoint['mn'], checkpoint['std']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch_start = checkpoint['epoch']
        # Generate data
        train_data, val_data, mn, std = utils.generate_training_data(n=20000, mn=mn, std=std)
    else:
        # Generate data
        train_data, val_data, mn, std = utils.generate_training_data(n=20)

    data_train_loader = DataLoader(train_data, batch_size=256, shuffle=True, num_workers=0)
    data_test_loader = DataLoader(val_data, batch_size=1284, shuffle=True, num_workers=0)
    criterion = DIOULOSS(mn=mn, std=std, w=200, h=200)  # nn.MSELoss()

    if not epoch_start:
        epoch_start = 1
    for e in range(epoch_start, 3000):
        train_and_test(epoch=e, model=model, criterion=criterion, optimizer=optimizer,
                       data_train_loader=data_train_loader, data_test_loader=data_test_loader, mn=mn, std=std )


if __name__ == '__main__':
    main()