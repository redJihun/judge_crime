# tensorflow & keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical, normalize

# for graph
import matplotlib.pyplot as plt



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
