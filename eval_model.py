import time
import sys,os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from data_loader import TurnDataset
from model import TurnTakingLSTM

import visdom

feat_size = 52 #13, 40
hidden_dim = 32 #64
input_len = 20 # 100
pred_len = 10  # 50
epoch_num = 1
batch_size = 10 #16, 32, 64

model_path = './model.pth'

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

# visualize using visdom
vis_test = visdom.Visdom() 
vis_acc = visdom.Visdom() 
plot_test = vis_test.line(Y=torch.tensor([0]), X=torch.tensor([0]))
plot_acc = vis_acc.line(Y=torch.tensor([0]), X=torch.tensor([0]))

def eval(model, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data, target in test_loader:
            # print('target val: {}'.format(target))
            data = data.to(device)
            target = target.to(device)
            output = model(data)

            test_loss = criterion(output, target)
            # print('target shape: {}, output shape: {}, test loss:{}'.format(target.shape, output.shape, test_loss))

            pred = output.data.max(0)[0]
            target_max_idx = target.data[0]
            pred_max_idx = pred.long()
            # print('target:: {}, prediction:: {}'.format(target_max_idx, pred_max_idx))

            correct += target_max_idx.eq(pred_max_idx).sum()
            # correct += pred.eq(target.data).sum()
            print('correct: {}'.format(correct))

    accuracy = 100 * correct / (len(test_loader.dataset) * pred_len)
    print('\n[Test accuracy : {:.2f}%], ({}/{})'.format(accuracy, correct, len(test_loader.dataset)))

    test_loss /= len(test_loader.dataset)

    # test loss-> Y, epoch-> X
    vis_test.line(Y=[test_loss], X=np.array([epoch]), win=plot_test, update='append')
    vis_acc.line(Y=[accuracy], X=np.array([epoch]), win=plot_acc, update='append')

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("USAGE: eval_model.py <feature_path> <label_dataset_path>")
        exit(-1)
    else:
        feature_folder = sys.argv[1]
        label_folder = sys.argv[2]

    # define test loader
    test_dataset = TurnDataset(feature_folder, label_folder,
                         input_length=input_len, prediction_length=pred_len, is_train=False)
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # define model
    model = TurnTakingLSTM(feat_size, hidden_dim, pred_len, batch_size).to(device)
    model.load_state_dict(torch.load(model_path))

    # for epoch in range(epoch_num):
    epoch = 1
    eval(model, test_loader, epoch)


