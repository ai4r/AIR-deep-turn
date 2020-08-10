#-*- coding:utf-8 -*-
import csv
import pandas as pd
import sys
import os
import glob
import numpy as np

frame_times = 100  # 50ms
un_classified_label = 5

# extract turn-taking label from the elan generated turn-taking label
def readcsv(csv_path):
    # annotatation dsataset
    annotation_dir = csv_path + 'annotation/'
    if not(os.path.exists(annotation_dir)):
        os.mkdir(annotation_dir)

    filelist = sorted(glob.glob(csv_path + "*.csv"))
    print(filelist)

    for i, f_name in enumerate(filelist):
        annot = pd.read_csv(f_name, delimiter=',')
        print('File name: ', f_name)
        print(annot)
        
        idx = annot.size/7 
        # row index
        print('idx::', idx)
        # end_time = annot['end'][idx-1]
        _, tail = os.path.split(f_name)  # split dir & file_name
        # print('tail name', tail)

        # time_index = frame_times
        # audio_samples = end_time/frame_times + 1  # audio total size, 50ms
        # print('file_name: {}, size: {}, end_time: {}, audio_length: {}'.format(
        #     f_name, idx, end_time, audio_samples))

        # annotaiton_dataset reshape()
        annotation_data = np.array([['start','end','class']])
        print('annotation_data: {}'.format(annotation_data))
        # data_list = [['time_val', 'label']]
        # annotation_data = np.array(data_list)

        j = 0
        while j < idx:
            start_time = annot['start'][j]
            end_time = annot['end'][j]

            print('idx: {},start: {}, end: {}'.format(
                j, start_time, end_time))

            if annot['take'][j] == 'take':
                label = 0
            elif annot['release'][j] == 'release':
                label = 1
            elif annot['wait'][j] == 'wait':
                label = 2
            elif annot['hold'][j] == 'hold':
                label = 3
            elif annot['other_state'][j] == 'other_state':
                label = 4
            print('j, label value::',j, label)

            annotation_data = np.append(
                annotation_data, np.array([[start_time, end_time, label]]), axis=0)

            j += 1

            df = pd.DataFrame(annotation_data)
            df.to_csv(annotation_dir + tail, header=False, index=False)
            print('saved file name:', annotation_dir + tail)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("USAGE: data_prepare.py <label_path>")
        exit(-1)
    else:
        csv_folder = sys.argv[1]
        # wav_folder = sys.argv[1]

    # extract_wav_info(wav_folder)
    readcsv(csv_folder)
