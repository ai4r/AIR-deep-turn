import time
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from model import TurnTakingLSTM
from data_loader import TurnDataset

feat_size = 60
hidden_dim = 64
input_len = 7
pred_len = 5
epoch_num = 250
batch_size = 128
lr = 0.001

model_path = './models/model.pth'

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

def train(model, train_loader, optimizer, epoch_num):
    print('[Training... ]')
    losses = np.zeros(epoch_num)

    for epoch in range(epoch_num):
        start = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)

            model.zero_grad() # init gradient value
            model.hidden = model.init_hidden()

            optimizer.zero_grad()

            # forward pass
            output = model(data)

            loss = loss_function(output, target)

            loss.backward()

            # update weights
            optimizer.step()

            # record loss history
            losses[epoch] += loss.item()

            if batch_idx % 100 == 0:
                print('[epoch: {}, step: {}/{}, loss: {}]'.format(epoch+1, batch_idx, len(train_loader), loss.item()))

        losses[epoch] /= len(train_loader)
        train_losses.append(losses[epoch])

        print('[Epoch: {}, loss: {}, took {} sec]'.format(
            epoch+1, losses[epoch], time.time() - start))

        # save model
        torch.save(model.state_dict(), model_path)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("USAGE: train.py <features_path> <label_path>")
        exit(-1)
    else:
        feature_folder = sys.argv[1]
        label_folder = sys.argv[2]

    # define train_losses
    train_losses = []

    # train loader
    train_dataset = TurnDataset(feature_folder, label_folder,
                          input_length=input_len, prediction_length=pred_len, is_train=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                              shuffle=True, drop_last=True, num_workers=2)

    # define model
    model = TurnTakingLSTM(feat_size, hidden_dim,
                           pred_len, batch_size).to(device)

    loss_function = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr = lr, betas = (0.9, 0.999), eps = 1e-08, weight_decay = 0, amsgrad = False)
    # optimizer = optim.Adagrad(model.parameters(), lr=lr, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)
    step_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200], gamma=0.1)

    train(model, train_loader, optimizer, epoch_num)
