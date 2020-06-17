# for handle data
import numpy as np
import pandas as pd

# for load wav & mfcc
import os
from os import listdir
from os.path import isfile, join
import librosa

# preprocessing
from sklearn.preprocessing import MinMaxScaler
""" Load WAV files & ground truth"""
DIR_PATH = '.\\dataset'
SAMPLE_RATE = 22050

audio_files = [join(DIR_PATH, f) for f in listdir(DIR_PATH) if isfile(join(DIR_PATH, f)) if 'wav' in f]
audio_files.sort()
print('wave files : ', len(audio_files))

ground_truth = pd.read_csv('./labels.csv', header=None)
ground_truth = ground_truth.values
ground_truth = ground_truth.reshape(len(audio_files), 1)

""" Covert to MFCC & Preprocess data for RNN"""
N_MFCC = 20
file_features = np.empty((len(audio_files), N_MFCC * 431))
# print(type(file_features))
for index in range(len(audio_files)):
    print('index : %d' % index)
    audio_file, sr = librosa.core.load(audio_files[index], sr=SAMPLE_RATE)
    file_feature = librosa.feature.mfcc(y=audio_file, sr=SAMPLE_RATE, S=None, n_mfcc=N_MFCC)
    # file_feature = librosa.util.normalize(file_feature, norm=np.inf)
    file_feature = np.ravel(file_feature, order='F').reshape(1, -1)
    # file_features = np.append(file_features, file_feature, axis=0)
    file_features[index] = file_feature

# file_features += 1
file_features = MinMaxScaler(feature_range=(0, 1)).fit_transform(file_features)
print(file_features.shape)
print(file_features)
data = np.hstack((file_features, ground_truth))

""" Save the preprocessing data """
np.save('./np_data4', data)