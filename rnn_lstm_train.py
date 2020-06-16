# tensorflow & keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical, normalize

# for handle data
import numpy as np

# for graph
import matplotlib.pyplot as plt

# for load wav & mfcc
import os
from os import listdir
from os.path import isfile, join
import librosa

""" Load WAV files & ground truth"""
DIR_PATH = ''
SAMPLE_RATE = 22050

audio_files = [join(DIR_PATH, f) for f in listdir(DIR_PATH) if isfile(join(DIR_PATH, f)) if 'wav' in f]
audio_files.sort()
audio_file, sr = librosa.core.load(audio_files[0], sr=SAMPLE_RATE)

ground_truth = ''

""" Covert to MFCC """
N_MFCC = 256

file_feature = librosa.feature.mfcc(y=audio_file, sr=SAMPLE_RATE, S=None, n_mfcc=N_MFCC)

""" Make Model """
model = Sequential()
model.add(Embedding(N_MFCC, 100))
model.add(LSTM(100, activation='tanh'))
model.add(Dense(46, activation='softmax'))

""" Compile the Model """
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

""" Run the Model """
history = model.fit(audio_files, groud_truth, batch_size=100, epoch=20, validation_data=(x_test, y_test))

""" Test the Model """
print("\n 정확도 : %.4f" % (model.evaluate(x_test, y_test)[1]))

y_test_loss = history.history['val_loss']
y_train_loss = history.history['loss']

x_len = np.arange(len(y_test_loss))
plt.plot(x_len, y_test_loss, marker=',', c='red', label='Testset_loss')
plt.plot(x_len, y_train_loss, marker=',', c='blue', label='Trainset_loss')

plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
