import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
import matplotlib.pyplot as plt
import scipy.stats as stats
from fetch_functions import ad_line, adx, aroon_oscillator, stochastic_oscillator
import requests

    
#btc_data = pd.read_csv("./data/btc_big_data.csv")
# btc_close = btc_data["close"]
# btc_data.iloc[:-len(data)]
# data["btc_close"] = btc_close

    # Test stacionarity
    # def test_stationarity(timeseries):
    #     """
    #     Provede Augmented Dickey-Fuller test na danou časovou řadu.
    #     """
    #     print('Results of Dickey-Fuller Test:')
    #     dftest = adfuller(timeseries, autolag='AIC')
    #     dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    #     for key,value in dftest[4].items():
    #         dfoutput['Critical Value (%s)'%key] = value
    #     print(dfoutput)


# Předzpracování
def preprocess(data):
    data["p"] = (data["close"].diff() > 0).astype(int)
    data["target"] = data["p"].shift(-1)
    data = data.dropna()


    data["close"] = data["close"].diff()
    data = data.dropna()

    data['AD Line'] = ad_line(data)
    data['ADX'] = adx(data)
    data = pd.concat([data, aroon_oscillator(data)], axis=1)
    data[['SlowK', 'SlowD']] = stochastic_oscillator(data)
    data["log_vol"] = np.log(data["volume"])
    data = data.dropna()
    return data


# for col in ["close", "volume", "rsi_14", "ema_10", "ema_50", "atr", "btc_close"]: # přidal jsem btc_close
#     print(f"Test stacionarity pro {col}:")
#     test_stationarity(data[col])
#     print("-" * 20)
# Vytvoření tréninkových dat

def logit_model(data_input):
    
    data = preprocess(data_input)
    
    x = data[["atr","rsi_14", "ema_10", 'AD Line','SlowK', 'ADX',"Aroon Oscillator" , "log_vol"]].astype(np.float32)
    y = data["target"].values.astype(np.float32)

    x = sm.add_constant(x)
    x_train = x.iloc[:-1]
    y_train = y[:-1]
    x_test = x.iloc[-1:]

# Plotting the variables
# data[["volume","log_vol", "rsi_14", "ema_10", 'OBV_sol', 'AD Line','SlowK', 'SlowD', 'ADX', "Aroon Oscillator" ]].plot(subplots=True, figsize=(10, 12))
# plt.tight_layout()
# plt.show()
# Přidání konstanty (interceptu)


# Logistická regrese pomocí statsmodels
    model = sm.Logit(y_train, x_train)
    result = model.fit()

# Predikce
    y_pred = result.predict(x_test) # 0.5 je běžný práh pro binární klasifikaci

    return y_pred

# Vytvoření matice záměn
# cm = confusion_matrix(y_test, y_pred)

# # Vytvoření přehledné tabulky
# cm_df = pd.DataFrame(cm, index=["Predicted Negative", "Predicted Positive"], 
#                      columns=["True Negative", "True Positive"])


# # Vypsání tabulky
# print("Confusion Matrix:")
# print(cm_df)

# # Alternativně můžete přímo vypsat jednotlivé hodnoty
# TN, FP, FN, TP = cm.ravel()
# print("\nConfusion Matrix values:")
# print(f"True Negatives (TN): {TN}")
# print(f"False Positives (FP): {FP}")
# print(f"False Negatives (FN): {FN}")
# print(f"True Positives (TP): {TP}")

# # Výpočet přesnosti modelu
# accuracy = accuracy_score(y_test, y_pred)
# print("Přesnost modelu:", accuracy)


# # Test for autocorrelation of residuals
# dw_statistic = durbin_watson(result.resid_response)
# print(f"Durbin-Watson statistic: {dw_statistic}")

# # Plot distribution of residuals
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.hist(result.resid_response, bins=30, edgecolor='k')
# plt.title('Histogram of Residuals')

# plt.subplot(1, 2, 2)
# stats.probplot(result.resid_response, dist="norm", plot=plt)
# plt.title('Q-Q Plot of Residuals')

# plt.tight_layout()
# plt.show()

# # Check for multicollinearity
# df = pd.DataFrame(X_train)
# datacamp_vif_data = pd.DataFrame()
# datacamp_vif_data['Feature'] = df.columns
# datacamp_vif_data['VIF'] = [variance_inflation_factor(df.values, i) for i in range(X_train.shape[1])]


# print("Variance Inflation Factor (VIF):")
# print(datacamp_vif_data)

