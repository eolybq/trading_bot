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
import requests
from config import cmc_key


session = HTTP(
    demo=True,
    api_key=apikey,
    api_secret=secret,
)


def get_balance():
    try:
        resp = session.get_wallet_balance(accountType="UNIFIED", coin="USDT")

        return resp
    except Exception as e:
        print(e)



def get_data(start_time, end_time, interval, maxdata=200):
    data = []
    while start_time < end_time:
        response = session.get_kline(
            category="spot",
            symbol="BTCUSDT",
            interval=interval,
            start=start_time,
            end=min(start_time + maxdata * 1000 * 60 * interval, end_time),
        )
        print(f"Staženo {len(response['result']['list'])} svíček")
        data.extend(response["result"]["list"])
        start_time += maxdata  * 1000 * 60 * interval
        # time.sleep(1)
        print(start_time, end_time)
    return data

def make_dataframe(start_time, end_time):
    data = get_data(start_time, end_time, 15)
    date = []
    open = []
    high = []
    low = []
    close = []
    volume = []
    turnover = []
    for i in data:
        date.append(i[0])
        open.append(i[1])
        high.append(i[2])
        low.append(i[3])
        close.append(i[4])
        volume.append(i[5])
        turnover.append(i[6])

    tabulka = pd.DataFrame(
        {
            "date": date,
            "open": open,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
    tabulka = tabulka.sort_values(by="date", ascending=True)
    return tabulka


def ema_model(close_series, span, min_periods):
    ema = close_series.ewm(span=span, adjust=False, min_periods=min_periods).mean()
    return ema


#rsi
def rsi_model(close_floats):
    rsi = ta.rsi(close_floats)
    return rsi


def atr_model(high, low, close_floats):
    data_horizontal = pd.DataFrame()
    high_floats = high.astype(float)
    low_floats = low.astype(float)
    data_horizontal["tr_0"] = abs(high_floats - low_floats)
    data_horizontal["tr_1"] = abs(high_floats - close_floats.shift())
    data_horizontal["tr_2"] = abs(low_floats - close_floats.shift())
    tr = data_horizontal[['tr_0', 'tr_1', 'tr_2']].max(axis=1)
    return tr




def wema(period, tr):
    alpha = 1 / (period + 1)
    atr_values = [tr[:period].mean()]
    for value in tr[period:]:
        atr = (value * alpha) + (atr_values[-1] * (1 - alpha))
        atr_values.append(atr)
    return atr_values

def obv(data, close_column):
    obv_values = [0]  # Začneme s počáteční hodnotou 0 pro OBV
    for i in range(1, len(data)):  # Začneme od indexu 1, protože potřebujeme porovnávat s předchozím
        if data[close_column].iloc[i] > data[close_column].iloc[i-1]:
            obv_values.append(obv_values[-1] + data['volume'].iloc[i])
        elif data[close_column].iloc[i] < data[close_column].iloc[i-1]:
            obv_values.append(obv_values[-1] - data['volume'].iloc[i])
        else:
            obv_values.append(obv_values[-1])
    
    return pd.Series(obv_values, index=data.index, name='obv')

    
def ad_line(data):
    clv = (data['close'] - data['low']) - (data['high'] - data['close']) / (data['high'] - data['low'])
    ad = clv * data['volume']
    return pd.Series(ad.cumsum(), index=data.index, name="AD Line")
def adx(data, period=14):
    plus_dm = data['high'].diff()
    minus_dm = -data['low'].diff()

    plus_dm = plus_dm.where(plus_dm > 0, 0)
    minus_dm = minus_dm.where(minus_dm > 0, 0)

    tr = atr_model(data['high'], data['low'], data['close'])

    plus_di = 100 * (plus_dm.rolling(window=period).sum() / tr)
    minus_di = 100 * (minus_dm.rolling(window=period).sum() / tr)

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period, min_periods=period).mean()

    return pd.Series(adx, index=data.index, name="ADX")
def aroon_oscillator(data, period=14):
    aroon_up = ((data['high'].rolling(window=period).apply(lambda x: x.argmax())) / period) * 100
    aroon_down = ((data['low'].rolling(window=period).apply(lambda x: x.argmin())) / period) * 100
    aroon_osc = aroon_up - aroon_down
    return pd.Series(aroon_osc, index=data.index, name="Aroon Oscillator")
def stochastic_oscillator(data, k_period=14, d_period=3):
    highest_high = data['high'].rolling(window=k_period).max()
    lowest_low = data['low'].rolling(window=k_period).min()

    slowk = 100 * (data['close'] - lowest_low) / (highest_high - lowest_low)
    slowd = slowk.rolling(window=d_period).mean()

    # Combine slowk and slowd into one series
    stoch_osc = pd.concat([slowk, slowd], axis=1)
    stoch_osc.columns = ['SlowK', 'SlowD']
    return stoch_osc

