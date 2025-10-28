import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import math
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

tf.config.set_logical_device_configuration(
    physical_devices[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4072)])

# Funkce pro vytvoření sekvencí z dat
def seq_maker(data, sequence, predicted_feature):
    x, y = [], []
    for i in range(len(data) - sequence):
        x.append(data[i:i + sequence])
        y.append(data[i + sequence, predicted_feature])
    return np.array(x), np.array(y)

# trénink na všech historických datech
def rnn_train_model(data, seq_len, predicted_feature, batch_size, epochs):


    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(data)

    joblib.dump(scaler, './models/scaler_rnn_bt_model.pkl')

    x_train, y_train = seq_maker(train_scaled, seq_len, predicted_feature)

    
    # Define the RNN model
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(1500, activation='tanh', input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.LSTM(1000, activation='tanh'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(1))


    # Compile the model
    model.compile(optimizer = 'adam',
                        loss = 'mean_squared_error',
                        metrics = ["accuracy"])

    # Summary of the model
    model.summary()

    checkpoint = keras.callbacks.ModelCheckpoint(filepath='./models/rnn_bt_model_checkpoint.keras', 
                            save_best_only=True, 
                            monitor='val_loss', 
                            mode='min')
    
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(x_train, y_train, shuffle = False, batch_size=batch_size, epochs=epochs, validation_split=0.2, callbacks=[checkpoint, early_stopping])

    joblib.dump(history.history, './models/train_history.pkl')


    model.save('./models/rnn_bt_model.keras')

# Predikce pomocí nových dat
def rnn_predict(new_data, seq_len, number_features, predicted_feature):
    scaler = joblib.load('./models/scaler_rnn_bt_model.pkl')

    model = keras.models.load_model('./models/rnn_bt_model.keras')

 
    # data = new_data.values.reshape(-1, 1)
    # print(new_data.shape, data.shape)

    scaled_new = scaler.transform(new_data)

    rs_new = np.array(scaled_new).reshape(-1, seq_len, number_features)

    predictions = model.predict(rs_new)

    # Denormalizace predikcí
    # predictions_rescaled = predictions.reshape(-1, 1)  # Predikce mají tvar (n, 1)

    zeros_matrix_pred = np.zeros((predictions.shape[0], number_features))

    zeros_matrix_pred[:, predicted_feature] = predictions.flatten() 

    # Inverzní transformace
    predictions_inverse = scaler.inverse_transform(zeros_matrix_pred)[:, predicted_feature] # Denormalizace pouze sloupce "close"


    return predictions_inverse
