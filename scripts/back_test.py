from rnn import rnn_train_model, rnn_predict
import pandas as pd
import numpy as np
from datetime import datetime
from arima import arima_model
from logit import logit_model
import os
import time
import matplotlib.pyplot as plt
import warnings

# Potlačení všech varování
warnings.filterwarnings("ignore")


data = pd.read_csv("./data/sol_big_data_tf15.csv")
data = data.dropna()

train_data = data.iloc[:int(len(data) * 0.99995982)]
backtest_data = data.iloc[int(len(data) * 0.99995982):]

print(backtest_data)

# Trenink RNN
# Cesta k souboru
rnn = "models/rnn_bt_model.keras"

if not os.path.isfile(rnn):
    rnn_train_model(train_data.iloc[:, 4:].values, seq_len=64, predicted_feature=-1, batch_size=32, epochs=20)



test_df = pd.DataFrame(columns=["RNN", "ARIMA", "Naive", "Logit", "Real", "Change", "coeficient", "signal"])

# main cyklus
for i in range(1, len(backtest_data)):
    start_iter_time = time.time()

    preds = []


    coeficient = 0


    rnn_prediction = 0
    # RNN
    if i > 64:
        data_for_rnn = backtest_data.iloc[i-64:i, 4:].values
        rnn_prediction = rnn_predict(new_data=data_for_rnn, seq_len=64, number_features=data_for_rnn.shape[1], predicted_feature=-1)

        print(f"RNN: {rnn_prediction}, real: {backtest_data.iloc[i, 4]}")



        if rnn_prediction > 0:
            coeficient += 2

    preds.append(rnn_prediction)

    next_row = backtest_data.iloc[:i]
    data_merged = pd.concat([train_data, next_row], axis=0)

    # ARIMA
    arima_pred_temp = arima_model(data_merged)
    arima_pred = arima_pred_temp.iloc[0]
    naive_pred = data_merged["close"].iloc[-1]

    preds.append(arima_pred)
    preds.append(naive_pred)

    if arima_pred > 0:
        coeficient += 4
    
    # LOGIT
    logit_pred = logit_model(data_merged).iloc[0]
    preds.append(logit_pred)


    if logit_pred >= 0.515:
        coeficient += 2

    coef_flag = 1 if coeficient > 10 else 0

    #GOLDEN CROSS
    if backtest_data['ema_10'].iloc[i] > backtest_data['ema_50'].iloc[i] and backtest_data['ema_10'].iloc[i-1] <= backtest_data['ema_50'].iloc[i-1]:
        coeficient += 1
    #OVERSOLD
    if backtest_data["rsi_14"].iloc[i] < 30 and backtest_data['rsi_14'].iloc[i-1] >= 30:
        coeficient += 1

    signal = ""
    if coeficient > 6:
        signal = "BUY"

    test_df.loc[i] = [preds[0], preds[1], preds[2], preds[3], backtest_data.iloc[i, 4], backtest_data.iloc[i, -1], coeficient, signal]

    end_iter_time = time.time()

    # Výpočet času trvání této iterace
    elapsed_iter_time = end_iter_time - start_iter_time

    # Vypisování času pro tuto iteraci
    print(f"Iterace {i}/{len(backtest_data)} - Čas: {elapsed_iter_time:.4f} sekund")
    

fig, ax = plt.subplots(1, 2, figsize=(15, 10))
ax[0].plot(test_df["Real"], label="Real", c="b")
ax[0].plot(test_df["ARIMA"], label="ARIMA", c="g")
ax[0].plot(test_df["Naive"], label="Naive", c="y")
ax[1].plot(test_df["RNN"], label="RNN", c="r")
ax[1].plot(test_df["Change"], label="Change", c="violet")
ax[1].plot(test_df["Logit"], label="Logit", c="m")
coef_flag_indices = test_df[test_df["coeficient"] == 6].index
plt.scatter(coef_flag_indices, test_df.loc[coef_flag_indices, "Real"], color='red', marker='o')
ax[0].legend()
ax[1].legend()
plt.tight_layout()
plt.show()
print(test_df)