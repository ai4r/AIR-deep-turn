#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import sys
import librosa
import numpy as np

def extract_feature(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    # print('y shape: {}, sr: {}'.format(y.shape, sr))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_fft=800, hop_length=800, n_mfcc=40)
    # print('mfcc: {}, shape:{}'.format(mfcc, mfcc.shape))

    chroma = librosa.feature.chroma_stft(
        y=y, sr=sr, n_fft=800, hop_length=800)
    # print('Chroma: {}, shape:{}'.format(chroma, chroma.shape))

    rms= librosa.feature.rms(y=y, hop_length=800)
    # print('RMS: {}, shape:{}'.format(rms, rms.shape))

    tonnetz = librosa.feature.tonnetz(y=y, sr=sr, chroma=chroma)
    # print('tonnetz: {}, shape:{}'.format(tonnetz, tonnetz.shape))

    onset_feat = librosa.onset.onset_strength(y=y, sr=sr, hop_length=800, n_fft = 800)
    onset_feat = onset_feat.reshape(1, -1)
    # print('onset_feat: {}, shape:{}'.format(onset_feat, onset_feat.shape))

    feat = np.concatenate((mfcc, chroma, rms, tonnetz, onset_feat), axis=0)
    print('Features shape: {}'.format(feat.shape))
    return feat

def extract_features(path):
    feature_dir = path + 'feature_res/'

    if not(os.path.exists(feature_dir)):
        os.mkdir(feature_dir)

    for r, d, fnames in os.walk(path):
        for f in fnames:
            if not f.endswith('.wav'):
                continue
            fpath = os.path.join(r, f)

            print("Processing: ", fpath)
            feature = extract_feature(fpath)
            print('shape', feature.shape)

            # df = pd.DataFrame(feature)
            # df.to_csv(feature_dir + f + 'b.csv', header=False, index=True)

            if feature is not None:
                print("feature shape: ", feature.shape)
                np.save(feature_dir + f + ".npy", feature)

            else:
                print("Error: No Feature Extracted.")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("USAGE: feature_extraction.py <data_source>")
        exit(-1)
    else:
        data_folder = sys.argv[1]
    extract_features(data_folder)
