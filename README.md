# AIR-deep-turn
Multimodal turn-taking using deep learning

## Requirements:
- PyTorch ver > 1.4.0

Downloads:
- Download the git file
  - git clone https://github.com/ai4r/AIR-deep-turn.git

## Train & Evaludation
Train & evaluation model:

  ```
  python train.py <feature_path> <label_dataset_path>
  ```

  ```
  python eval_model.py <feature_path> <label_dataset_path>
  ```

## Feature Extraction
Features can be extracted by the following command:

  ```
  cd utils
  python feature_extraction.py <path_to_folder_where_wave_files_are_saved>
  ```

For each wave file, a feature file is created with the wave file name and an extension of '.npy' under the same folder.

## E-MIC(Elderly Multimodal Interpersonal Conversation) dataset

Downloads:
- Dataset
  - E-MIC Dataset: https://nanum.etri.re.kr:8080/etriPortal/share/view?id=31

## Open Dataset: The HCRC Map Task Corpus Dataset
Requirements:
- nltk: sudo apt-get install python-nltk or sudo pip install -U nltk
- Sox: sudo apt-get install sox
- OpenSmile-2.3.0

Downloads:
- Dataset
  - HCRC Map Task Corpus: http://groups.inf.ed.ac.uk/maptask/index.html
  - Download the maptask corpus audio data from http://groups.inf.ed.ac.uk/maptask/maptasknxt.html
  - Download opensmile from https://audeering.com/technology/opensmile/#download and extract into xanadu/utils. (uploaded...)

Preprocessing:
- Split the audio channels
  - sh preprocessing/split_channel.sh
- Extract audio features
  -  python preprocessing/extract_gemaps.py
  -  python preprocessing/prepare_gemaps.py
- Extract voice activity
  -  python preprocessing/get_va_annotation.py
- Find pause and extract hold/shift
  -  python preprocessing/find_pauses.py

