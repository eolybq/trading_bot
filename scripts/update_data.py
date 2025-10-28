import tensorflow as tf
import pandas as pd
from config import apikey, secret
from pybit.unified_trading import HTTP
from datetime import datetime, timezone
import csv
import numpy as np
import pandas_ta as ta
from fetch_functions import get_balance, get_data, ema_model, rsi_model, atr_model, wema, make_dataframe
from time import time
import matplotlib.pyplot as plt
import joblib


session = HTTP(
    demo=True,
    api_key=apikey,
    api_secret=secret,
)

data = pd.read_csv("./data/btc_big_data_tf15.csv")


start_time = data.iloc[-1, 0] + 300000
end_time = int(datetime.now().timestamp() * 1000)
tabulka = make_dataframe(start_time, end_time)

#updated_data = pd.concat([data, tabulka], axis=0)

historical_data = data.iloc[-50:].copy()
temp_data = pd.concat([historical_data, tabulka], axis=0)

close_series_temp = pd.Series(temp_data["close"])
close_floats_temp = temp_data["close"].astype(float)

temp_data["ema_10"] = ema_model(close_series_temp, 10, 10)
temp_data["ema_50"] = ema_model(close_series_temp, 50, 50)
temp_data["rsi_14"] = rsi_model(close_floats_temp)
temp_data["atr"] = pd.Series(wema(14, atr_model(temp_data["high"], temp_data["low"], close_floats_temp)))

updated_tabulka = temp_data.iloc[len(historical_data):]
updated_data = pd.concat([data, updated_tabulka], axis=0)

updated_data = updated_data.dropna()
updated_data["change"] =  pd.to_numeric(updated_data["close"], errors='coerce').pct_change()

updated_data.to_csv("./data/btc_big_data_tf15.csv", index=False)