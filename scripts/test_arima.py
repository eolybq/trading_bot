import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
import time


window_size = 20000  # Rolling window s velikostí 10

# Načtení dat
data = pd.read_csv("./data/updated_data.csv")
data = data.iloc[-20111:]
data = data["close"]

# Inicializace velikosti okna


# Uložení predikcí a skutečných hodnot pro ARIMA a naivní model
arima_predictions = []
naive_predictions = []
actual_values = []

# Rolling window predikce
for i in range(window_size, len(data)):
    start_time = time.time()

    # Tréninková sada pro aktuální okno
    train_window = data.iloc[i-window_size:i]
    
    # Trénování ARIMA modelu na aktuálním okně
    model = sm.tsa.ARIMA(train_window, order=(30, 1, 0))
    results = model.fit()
    
    # ARIMA predikce pro příští krok
    arima_pred = results.forecast(steps=1)
    arima_predictions.append(arima_pred.iloc[0])

    # Naivní model: předpověď = poslední známá hodnota
    naive_pred = train_window.iloc[-1]
    naive_predictions.append(naive_pred)
    
    # Uložení skutečné hodnoty
    actual_values.append(data.iloc[i])

    # if i % 100 == 0:
    print(f"Predikce ARIMA: {arima_pred.iloc[0]} | Predikce Naivní: {naive_pred} | Aktuální hodnota: {data.iloc[i]}")
    print(f"Number: {i-window_size + 1} / {len(data) - window_size}")


    end_time = time.time()  # Konec měření času
    print(f"Iterace {i+1} trvala {end_time - start_time} sekund")

# Výpočet metrik pro ARIMA model
arima_mae = mean_absolute_error(actual_values, arima_predictions)
arima_rmse = np.sqrt(mean_squared_error(actual_values, arima_predictions))

# Výpočet metrik pro Naivní model
naive_mae = mean_absolute_error(actual_values, naive_predictions)
naive_rmse = np.sqrt(mean_squared_error(actual_values, naive_predictions))

print("\n--- ARIMA Model ---")
print(f"Mean Absolute Error (MAE): {arima_mae}")
print(f"Root Mean Squared Error (RMSE): {arima_rmse}")

print("\n--- Naivní Model ---")
print(f"Mean Absolute Error (MAE): {naive_mae}")
print(f"Root Mean Squared Error (RMSE): {naive_rmse}")

# Vykreslení skutečných hodnot vs. predikce
plt.figure(figsize=(12, 6))
plt.plot(range(window_size, len(data)), actual_values, label="Skutečné hodnoty", color="blue")
plt.plot(range(window_size, len(data)), arima_predictions, label="Predikce ARIMA", color="red")
plt.plot(range(window_size, len(data)), naive_predictions, label="Predikce Naivní", color="green")
plt.legend()
plt.title("Porovnání ARIMA a Naivního modelu")
plt.show()
