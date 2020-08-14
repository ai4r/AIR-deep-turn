import glob

import torch
import numpy as np
import pandas as pd
import os

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class TurnDataset(Dataset):
    def __init__(self, feat_path, annotation_path, input_length, prediction_length, is_train):
        self.input_lengths = input_length
        self.prediction_length = prediction_length

        # feature & label list
        self.f_list = sorted(glob.glob(feat_path + "*.npy"))
        self.annot_list = sorted(glob.glob(annotation_path + "*.csv"))

        # train/test split
        train_size = int(len(self.f_list) ) #* 0.7)
        if is_train:  # train set
            self.f_list = self.f_list[0:train_size]
        else:  # test set
            self.f_list = self.f_list[train_size:]

        # read files
        feat_list = []
        annotation_list = []
        for i, (f_name, annotation_name) in enumerate(zip(self.f_list, self.annot_list)):
            feat_list.append(np.load(f_name).transpose())  # shape (n_frames, feat_size)
            _, tail = os.path.split(annotation_name)

            annotation_file = pd.read_csv(annotation_path + tail, delimiter=',')
            print('annotation_file shape: {}'.format(annotation_file.shape))

            annotation_list.append(annotation_file['label'].values)
            # print('annotation_values: {}'.format(annotation_file['label'].values))

        # make samples
        self.inputs = []
        self.labels = []
        for i, feat_f in tqdm(enumerate(feat_list)):
            predict_f = np.array([np.roll(annotation_list[i], -roll_idx) for roll_idx in range(1, prediction_length + 1)]).transpose()  # shape (frames, prediction_length)
            if (feat_f.shape[0] == predict_f.shape[0]):
                n_frames = feat_f.shape[0]
            else:
                n_frames = min(feat_f.shape[0], predict_f.shape[0])
            print('feat_f.shape[0]: {}, predict_f.shape[0]: {},frame num: {}'.format(feat_f.shape[0], predict_f.shape[0], n_frames))

            for j in range(input_length, n_frames, 5):
                input_f = feat_f[j - input_length:j]
                # output_f = predict_f[j - input_length:j]  # shape (seq_length, prediction_length)
                # output_f = predict_f[i:j]  # shape (seq_length, prediction_length)
                output_f = annotation_list[i][j:j+prediction_length]

                if (len(output_f) == prediction_length):
                    self.inputs.append(input_f)
                    self.labels.append(output_f)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # label_val = self.labels[idx][0][1].astype(int)
        # label_item = np.zeros((self.input_lengths, self.prediction_length))
        # if label_val and (label_val >= 0 and label_val < 6):
        #     label_item[:, label_val] = 1.
        # else:
        #     pass
        # print('idx: {}, label value: {}'.format(idx, label_val))


        return torch.from_numpy(self.inputs[idx]).float(), torch.from_numpy(self.labels[idx]).long()
        # return torch.from_numpy(self.inputs[idx]).float(), label_item.astype(np.float32)
