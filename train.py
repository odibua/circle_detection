from model import Net, DIOULOSS
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import utils


def train(epoch: int, model: nn.Module, criterion: nn.Module, optimizer: torch.optim, dataloader: torch.utils.data, mn: np.ndarray, std: np.ndarray):
    # Set model to train
    model.train()
    f = open("checkpoints/train_logs.txt", "a+")
    for i, (labels, images) in enumerate(dataloader):
        # Evaluate model
        optimizer.zero_grad()
        labels = torch.stack(labels).T.float()
        output = model(images.float())

        # Evaluate loss and iou
        loss, iou = criterion(output, labels)

        if i % 10 == 0:
            print('Train - Epoch %d, Batch: %d, Loss: %f IOU %f' % (epoch, i, loss.detach().cpu().item(), iou.detach().cpu().item()))
            print('Prediction: {pred} Label: {lab}'.format(pred=output[0:2], lab=labels[0:2]))

        # Backprop loss and optimization
        loss.backward()
        optimizer.step()
    # Save model, optimizer, and normalization constant every 5 epoches
    if epoch % 5 == 0:
        torch.save(
            {'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'epoch': epoch + 1,
             'std': std,
             'mn': mn,
             },
            f"checkpoints/model_epoch_{epoch}_batch_{i}")
        f.write(f"{epoch}, {i}, {loss.detach().cpu().item()}, {iou} \n")


def main():
    # Boolean to use checkpoint as initial model for training
    load_checkpoint = False
    utils.create_log_files("output.txt", 'epoch, batch, loss, iou\n')

    # Initialize model, loss function, and optimizer
    model = Net().float()
    optimizer = optim.Adadelta(model.parameters())

    epoch_start, mn, std = None, None, None
    if load_checkpoint:
        # Load checkpoint, epoch, model, optimizer and relevant params
        checkpoint = torch.load('checkpoints/model_epoch_100_batch_0')
        mn, std = checkpoint['mn'], checkpoint['std']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch_start = checkpoint['epoch'] + 1

        # Generate data
        train_data, val_data, _, _ = utils.generate_training_data(n=10000, mn=mn, std=std)
    else:
        # Generate data
        train_data, val_data, mn, std = utils.generate_training_data(n=10000)

    data_train_loader = DataLoader(train_data, batch_size=256, shuffle=True, num_workers=0)
    criterion = DIOULOSS(mn=mn, std=std, w=200, h=200)

    if not epoch_start:
        epoch_start = 1
    for e in range(epoch_start, 3000):
        train(epoch=e, model=model, criterion=criterion, optimizer=optimizer,
             dataloader=data_train_loader, mn=mn, std=std)


if __name__ == '__main__':
    main()