import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_loader import TurnDataset
from model import TurnTakingLSTM

feat_size = 60
hidden_dim = 64
input_len = 7
pred_len = 5
batch_size = 128

model_path = './models/model.pth'

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

def eval(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            in_data, target = data
            in_data = in_data.to(device)
            target = target.to(device)
            output = model(in_data)

            loss = criterion(output, target)

            _, predicted = torch.max(output.data, 1)
            correct += (predicted == target).sum().item()
            test_loss += loss.item()

        accuracy = 100 * correct / (len(test_loader.dataset))
        print('\n[Accuracy: {:.2f}%], ({}/{})'.format(accuracy, correct, len(test_loader.dataset)))

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("USAGE: eval_model.py <features_path> <label_path>")
        exit(-1)
    else:
        feature_folder = sys.argv[1]
        label_folder = sys.argv[2]

    accuracy_list = []
    test_losses = []
    accuracy = []

    # define test loader
    test_dataset = TurnDataset(feature_folder, label_folder,
                         input_length=input_len, prediction_length=pred_len, is_train=False)
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # define model
    model = TurnTakingLSTM(feat_size, hidden_dim, pred_len, batch_size).to(device)
    model.load_state_dict(torch.load(model_path))

    eval(model, test_loader)
