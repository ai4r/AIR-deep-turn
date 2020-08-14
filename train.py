import time
import sys,os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
# import matplotlib.pyplot as plt

from model import TurnTakingLSTM
from data_loader import TurnDataset

feat_size = 40  # 13, 40, 52
hidden_dim = 40 # 32
input_len = 10 # 20, 5sec
pred_len = 5  # 10, 2.5sec
epoch_num = 100
batch_size = 64
lr = 0.001

model_path = 'model/model.pth'

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
# print('device : {}'.format(device))

def train(model, train_loader, optimizer, epoch_num):
    print('[Training... ]')
    losses = np.zeros(epoch_num)

    for epoch in range(epoch_num):
        start = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            # print('batch idx: {},  target: {}'.format(batch_idx, target))
            data = data.to(device)
            target = target.to(device)
            # print('data shape: {}, target shape: {}'.format(data.shape, target.shape))

            model.zero_grad() #init gradient value
            model.hidden = model.init_hidden()

            optimizer.zero_grad()

            # forward pass
            output = model(data)
            # print('target: {}'.format(target))

            loss = loss_function(output, target)

            loss.backward()

            # update weights
            optimizer.step()

            # record loss history
            losses[epoch] += loss.item()

            if batch_idx % 100 == 0:
                print('[epoch: {}, step: {}/{}, loss: {}]'.format(epoch+1, batch_idx, len(train_loader), loss.item()))

        losses[epoch] /= len(train_loader)

        print('[Epoch: {}, loss: {}, took {} sec]'.format(
            epoch+1, losses[epoch], time.time() - start))  #epoch+1


    #save train model
    torch.save(model.state_dict(), model_path)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("USAGE: train.py <audio_feature_path> <label_dataset_path>")
        exit(-1)
    else:
        feature_folder = sys.argv[1]
        label_folder = sys.argv[2]

    # define train loader
    train_dataset = TurnDataset(feature_folder, label_folder,
                          input_length=input_len, prediction_length=pred_len, is_train=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                              shuffle=True, drop_last=True, num_workers=2) # num_workers=2

    # define test loader
    # test_dataset = TurnDataset(feature_folder, label_folder,
    #                      input_length=input_len, prediction_length=pred_len, is_train=False)
    # test_loader = DataLoader(
    #     dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # define model
    model = TurnTakingLSTM(feat_size, hidden_dim,
                           pred_len, batch_size).to(device)

    loss_function = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    train(model, train_loader, optimizer, epoch_num)

    # for epoch in range(1, epoch_num + 1):
    # for epoch in range(epoch_num):
    #     train(model, train_loader, optimizer, epoch)

    # #save train model
    # torch.save(model.state_dict(), model_path)
