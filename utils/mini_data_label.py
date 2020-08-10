import csv
import pandas as pd
import sys, os
import glob
import numpy as np

frame_times = 50  # 100ms
un_classified_label = 5


# def extract_wav_info(dataset_path):
#     wave_files = sorted(glob.glob(dataset_path +'*.wav'))

#     for f in enumerate(wave_files):
#         # f = sf.SoundFile(wave_f)
#         print ('file_name: {}'.format(f))
#         print('samples = {}'.format(len(f)))
#         print('sample rate = {}'.format(f.samplerate))
#         print('seconds = {}'.format(len(f) / f.samplerate))

def readcsv(csv_path):
    # annotatation dsataset
    annotation_dir = csv_path + 'annotation/'
    if not (os.path.exists(annotation_dir)):
        os.mkdir(annotation_dir)

    filelist = sorted(glob.glob(csv_path + "*.csv"))
    print(filelist)

    for i, f_name in enumerate(filelist):
        annot = pd.read_csv(f_name, delimiter=',')
        idx = annot.size/2 # /3  # row index
        print('row index:', idx)
        end_time = annot['time'][idx - 1]
        _, tail = os.path.split(f_name)  # split dir & file_name
        #
        time_index = frame_times
        # audio_samples = end_time / frame_times + 1  # audio total size, 50ms
        # print('file_name: {}, size: {}, end_time: {}, audio_length: {}'.format(f_name, idx, end_time, audio_samples))
        #
        annotation_data = np.array([['time', 'label']])  # annotaiton_dataset reshape()
        # print('annotation_data: {}'.format(annotation_data))


        j = 0
        while j < idx:
            start_time = annot['time'][j]
            label = annot['label'][j]
            if label == 0 or label == 2:
                label = 5
            else:
                label = label
            print('idx: {},start: {}, label: {}'.format(j, start_time, label))

            # to do...
            # while time_index <= end_time:
            annotation_data = np.append(annotation_data, np.array([[time_index, label]]), axis=0)
            time_index += frame_times

            # print('time_index: {}, end_time: {}, annotation_data: {}'.format(time_index, end_time, annotation_data))

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
