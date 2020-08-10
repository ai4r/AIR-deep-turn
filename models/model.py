import time
import sys,os

import torch
import torch.nn as nn
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
# print('using device {}'.format(device))

# a turn-taking lstm model
class TurnTakingLSTM(nn.Module):
    def __init__(self, feat_size, hidden_dim, target_size, batch_size):
        """

        :type hidden_dim: object
        """
        super(TurnTakingLSTM, self).__init__()
        self.num_hidden_layers = 1
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(feat_size, hidden_dim, num_layers=self.num_hidden_layers, batch_first=True)
        self.hidden2output = nn.Linear(hidden_dim, target_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.zeros(self.num_hidden_layers, self.batch_size, self.hidden_dim).to(device),
                torch.zeros(self.num_hidden_layers, self.batch_size, self.hidden_dim).to(device))

    def forward(self, feat):
        lstm_out, self.hidden = self.lstm(feat, self.hidden)
        fc_out = self.hidden2output(lstm_out)
        predictions = torch.sigmoid(fc_out)
        return predictions
