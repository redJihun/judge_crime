# tensorflow & keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical, normalize
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers

# for graph
import matplotlib.pyplot as plt

# for handle data
import numpy as np

# N_MFCC = 20
data = np.load('np_data4.npy')
np.random.shuffle(data)

DATA_ROW = int(data.shape[0])
DATA_COL = int(data.shape[1])
SPLIT_SIZE = int(DATA_ROW/10)
train_size = SPLIT_SIZE*8

# X_train, Y_train = np.array([data[:train_size, :-1]]), np.array([data[:train_size, -1]])
# X_val, Y_val = np.array([data[train_size:val_size, :-1]]), np.array([data[train_size:val_size, -1]])
# X_test, Y_test = np.array([data[val_size:, :-1]]), np.array([data[val_size:, -1]])
X_train, Y_train = data[:train_size, :-1], data[:train_size, -1]
X_test, Y_test = data[train_size:, :-1], data[train_size:, -1]

""" Make Model """
model = Sequential()
feature_size = DATA_COL-1
model.add(Embedding(feature_size, 100))
model.add(LSTM(100, activation='tanh', return_sequences=True))
model.add(LSTM(50, activation='tanh', return_sequences=False))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='softmax'))
adam = optimizers.Adam(lr=0.01, epsilon=1e-08, decay=1e-6)

""" Compile the Model """
model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

model.summary()

""" Run the Model """
checkpoint_path = "./training_log/cp5.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True, verbose=1)
history = model.fit(X_train, Y_train, batch_size=5, epochs=1,
                    validation_data=(X_test, Y_test), callbacks=[cp_callback])

""" Test the Model """
model.save('./training_log/model_5.h5')
print("\n 정확도 : %.4f" % (model.evaluate(X_test, Y_test)[1]))

y_test_loss = history.history['val_loss']
y_train_loss = history.history['loss']

# x_len = np.arange(len(y_test_loss))
# plt.plot(x_len, y_test_loss, marker=',', c='red', label='Testset_loss')
# plt.plot(x_len, y_train_loss, marker=',', c='blue', label='Trainset_loss')
#
# plt.legend(loc='upper right')
# plt.grid()
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.show()
