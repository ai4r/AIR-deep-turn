#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import sys
import librosa
import numpy as np

frame_length = 0.05 # window size: 50ms
frame_stride = 0.01 # stride: 10ms

def extract_feature(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    print('y shape: {}, sr: {}'.format(y.shape, sr))

    in_nfft = int(round(sr*frame_length)) # 800
    in_stride = int(round(sr*frame_stride)) # 160

    hop_length = 800
    # hop_length = 1103 #~= 50ms

    # raw features: 300 ~
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_fft=800, hop_length=800, n_mfcc=40) #hop_length = hop_length
    # normalized_mfcc = librosa.util.normalize(mfcc, axis=0, norm=1)
    # print('nomalized mfcc val:', normalized_mfcc)
    print('mfcc: {}, shape:{}'.format(mfcc, mfcc.shape))

    # chroma = librosa.feature.chroma_stft(
    #     y=y, sr=sr, n_fft=800, hop_length=800)
    # print('Chroma: {}, shape:{}'.format(chroma, chroma.shape))
    #
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length = 800)
    onset_feat = librosa.frames_to_time(onset_frames, sr=sr, hop_length=800, n_fft = 800)
    # print('onset_frame: {}, shape:{}'.format(onset_frames, onset_frames.shape))

    feat = np.concatenate((mfcc, onset_feat), axis=0)
    return feat

def extract_features(path):
    feature_dir = path + 'feature_res_50ms/'

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
            # print('csv file generated!!: {}'.format(df))

            if feature is not None:
                print("feature shape: ", feature.shape)
                np.save(feature_dir + f + ".npy", feature)

            else:
                print("Error: No Feature Extracted.")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("USAGE: feature_extraction.py <audio_data_source>")
        exit(-1)
    else:
        audio_folder = sys.argv[1]
    extract_features(audio_folder) 
