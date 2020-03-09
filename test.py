from model import Net, DIOULOSS
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import utils


def test(model: nn.Module, criterion: nn.Module, dataloader: torch.utils.data):
    model.eval()
    avg_loss, avg_iou = 0.0, 0.0
    results = []
    # import ipdb
    # ipdb.set_trace()
    for i, (labels, images) in enumerate(dataloader):
        labels = torch.stack(labels).T.float()
        output = model(images.float())
        loss, iou = criterion(output, labels)
        results.append(iou.detach().cpu().item())
        avg_loss += loss
        avg_iou += iou
        pred = output.detach()
    # ipdb.set_trace()
    avg_loss /= (i+1)
    avg_iou /= (i+1)
    results = np.array(results)

    # print(f"Test Avg. Loss: {avg_loss.detach().cpu().item()} Test Avg. IOU: {avg_loss.detach().cpu().item()}")
    return avg_loss, avg_iou, (results > 0.7).mean()


def main():
    utils.create_log_files("final_model/test_logs.txt", 'epoch, AP\n')
    f = open("final_model/test_logs.txt", "a+")

    # Initialize model, loss function, and optimizer
    model = Net().float()
    optimizer = optim.Adadelta(model.parameters())

    f_train = open("final_model/output.txt", "r")
    next(f_train)
    iou_list, acc_list = [],[]
    for line in f_train:
        meta = line.split(",")
        # Load checkpoint, epoch, model, optimizer and relevant params
        checkpoint = torch.load(f"final_model/model_epoch_{meta[0]}_batch_{meta[1].strip(' ')}")
        epoch = checkpoint['epoch']
        mn, std = checkpoint['mn'], checkpoint['std']
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        criterion = DIOULOSS(mn=mn, std=std, w=200, h=200, mn_label=0, std_label=1)
        N = 100
        acc = np.zeros((N,))

        for idx in range(N):
            # Generate data
            val_data, _, _, _ = utils.generate_training_data(n=500, mn=np.array([0, 0, 0]), std=np.array([1.0, 1.0, 1.0]))
            results = []
            for val in val_data:
                detected = utils.find_circle(model, val[1], mn, std)
                results.append(utils.iou(detected, val[0]))
            results = np.array(results)
            acc[idx] = (results > 0.7).mean()
        f.write(f"{epoch}, {acc.mean()} \n")
        print(f"{epoch}, {acc.mean()}, lower: {acc.mean() - 1.96 * acc.std()/(N**0.5)}, upper: {acc.mean() + 1.96 * acc.std()/(N**0.5)}")


if __name__ == '__main__':
    main()