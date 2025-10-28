import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
import time


def arima_model(data):

    window_size = 20000  # Rolling window s velikostí 10


    data = data.iloc[-20111:]
    close = data["close"]

    
    # Trénování ARIMA modelu na aktuálním okně
    model = sm.tsa.ARIMA(close, order=(30, 1, 0))
    results = model.fit()
    
    # ARIMA predikce pro příští krok
    arima_pred = results.forecast(steps=1)



    return arima_pred

