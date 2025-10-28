import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD
import os
import joblib


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Nastavení maximální alokace paměti na 4 GB (ponechání 2 GB pro systém)
tf.config.set_logical_device_configuration(
    physical_devices[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3096)])


os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'


data = pd.read_csv('./data/sol_big_data_tf15.csv')
# data["change"] = data["close"].pct_change()

data = data.dropna()

features = data.iloc[:, 4:].values



train_data, test_data = train_test_split(features, test_size=0.2, shuffle=False)


# # Normalize the data
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)


def seq_maker(close, sequence):
    x, y = [], []
    for i in range(len(close) - sequence):
        x.append(close[i:i + sequence])
        y.append(close[i + sequence, 4])
    return np.array(x), np.array(y)

sequence_length = 64

x_train, y_train = seq_maker(train_scaled, sequence_length)
x_test, y_test = seq_maker(test_scaled, sequence_length)

num_features = x_test.shape[2]


x_train = x_train.reshape(-1, sequence_length, num_features)
x_test = x_test.reshape(-1, sequence_length, num_features)


                             
model = keras.models.Sequential()
model.add(keras.layers.Input(shape=(x_train.shape[1], x_train.shape[2])))
model.add(keras.layers.LSTM(20, activation='tanh', return_sequences=True))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.LSTM(20, activation='tanh'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(1))

# Compile the model
model.compile(optimizer = 'adam',
                      loss = 'mean_squared_error',
                      metrics = ["accuracy"])

# Summary of the model
model.summary()

# early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = keras.callbacks.ModelCheckpoint(filepath='./models/big_model_checkpoint.keras', 
                             save_best_only=True, 
                             monitor='val_loss', 
                             mode='min')

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(x_train, y_train, shuffle = False, batch_size=32, epochs=5, validation_split=0.2, callbacks=[checkpoint, early_stopping])





predictions = model.predict(x_test)



# Denormalizace predikcí
predictions_rescaled = predictions.reshape(-1, 1)  # Predikce mají tvar (n, 1)
y_test_rescaled = y_test.reshape(-1, 1)  # Zajisti stejný tvar pro y_test

# Vytvoření matice nulových hodnot pro ostatní sloupce
num_features = test_scaled.shape[1]
zeros_matrix_pred = np.zeros((predictions.shape[0], num_features))
zeros_matrix_y = np.zeros((y_test.shape[0], num_features))

# Nahrazení příslušného sloupce predikcemi a y_test
zeros_matrix_pred[:, 4] = predictions.flatten()  # Sloupec odpovídá "close" (index 0 v sekvencích)
zeros_matrix_y[:, 4] = y_test.flatten()

# Inverzní transformace
predictions_inverse = scaler.inverse_transform(zeros_matrix_pred)[:, 4]  # Denormalizace pouze sloupce "close"
y_test_inverse = scaler.inverse_transform(zeros_matrix_y)[:, 4]

# Vizualizace
fig, ax = plt.subplots(2, 1, figsize=(12, 8))
ax[0].plot(y_test_inverse, label='Real Close Prices')
ax[0].plot(predictions_inverse, label='Predicted Close Prices')
ax[0].legend()

# ax[1].plot(history.history['loss'], label='Training loss')
# ax[1].plot(history.history['val_loss'], label='Validation loss')
ax[1].legend()

plt.savefig('./figs/rnn.png')
plt.show()
