from config import apikey, secret
from pybit.unified_trading import HTTP
from time import time
from datetime import datetime, timezone
import csv
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas_ta as ta
import matplotlib.dates as mdates
from fetch_functions import get_balance, get_data, ema_model, rsi_model, atr_model, wema, make_dataframe


session = HTTP(
    demo=True,
    api_key=apikey,
    api_secret=secret,
)

balance = get_balance()
print(f'Your balance is: {balance["result"]["list"][0]["totalEquity"]}')

start_time = int((datetime.now().timestamp() - 86400 * 1161) * 1000)
end_time = int(datetime.now().timestamp() * 1000)



tabulka = make_dataframe(start_time, end_time)



y = [float(item) for item in tabulka.close]
x = [float(item) for item in tabulka.date]

#EMA MODEL 10, 50
close_series = pd.Series(tabulka.close)
close_floats = tabulka["close"].astype(float)

tabulka["ema_10"] = ema_model(close_series, 10, 10)
tabulka["ema_50"] = ema_model(close_series, 50, 50)

tabulka["rsi_14"] = rsi_model(close_floats)

tabulka["atr"] = pd.Series(wema(14, atr_model(tabulka["high"], tabulka["low"], close_floats)))


atr_mean = tabulka["atr"].mean()
atr_std = tabulka["atr"].std()
low_vol = atr_mean - atr_std
high_vol = atr_mean + atr_std

tabulka.to_csv("./data/btc_big_data_tf15.csv", index=False)






sorted_indices = np.argsort(x)

x = np.array(x)
y = np.array(y)
x = x[sorted_indices]
y = y[sorted_indices]

x = pd.to_datetime(x, unit='ms')


fig, ax = plt.subplots(3, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1, 1]})

ax[0].plot(x, y, label='Reálná cena', color='blue')
ax[0].plot(x, tabulka["ema_10"], label='EMA 10', color='green')
ax[0].plot(x, tabulka["ema_50"], label='EMA 50', color='red')
ax[0].legend()
ax[0].grid(True)

ax[1].plot(x, tabulka["rsi_14"], label='RSI (14)', color='orange')
ax[1].axhline(70, color='red', linestyle='--', linewidth=1, label="Overbought") 
ax[1].axhline(30, color='green', linestyle='--', linewidth=1, label="Overselled")  
ax[1].set_title('RSI Indikátor')
ax[1].set_ylabel('RSI')
ax[1].set_xlabel('Čas')
ax[1].legend()
ax[1].grid(True)

ax[2].plot(x, tabulka["atr"], label='ATR', color='purple')
ax[2].set_title('ATR Indikátor')
ax[2].set_ylabel('ATR')
ax[2].axhline(y=low_vol, color='green', linestyle='--', linewidth=1, label="Nízká volatilita") 
ax[2].axhline(y=high_vol, color='red', linestyle='--', linewidth=1, label="Vysoká volatilita")  
ax[2].legend()
ax[2].grid(True)

for a in ax:
    a.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    a.xaxis.set_major_locator(mdates.HourLocator(interval=4))



plt.tight_layout()


plt.savefig("./figs/close_prices.png", dpi=300)