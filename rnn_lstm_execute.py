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

# tensorflow & keras
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical, normalize
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers

# for graph
import matplotlib.pyplot as plt

""" Load WAV files & ground truth"""
# DIR_PATH = '.\\record'
SAMPLE_RATE = 22050

file_name = './record/soundscape_unimodal4147.wav'
# file_path = [join(DIR_PATH, file_name)]

label = 1
ground_truth = [[label]]
""" Covert to MFCC & Preprocess data for RNN"""
N_MFCC = 20

audio_file, sr = librosa.core.load(file_name, sr=SAMPLE_RATE)
file_feature = librosa.feature.mfcc(y=audio_file, sr=SAMPLE_RATE, S=None, n_mfcc=N_MFCC)
file_feature = np.ravel(file_feature, order='F').reshape(1, -1)

file_feature = MinMaxScaler(feature_range=(0, 1)).fit_transform(file_feature)
print(file_feature.shape)
print(file_feature)
data = np.hstack((file_feature, ground_truth))

model = tf.keras.models.load_model('.\\training_log\\model_5.h5')
model.summary()
model.predict_classes(data)