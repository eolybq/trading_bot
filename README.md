# Trading bot

! THIS PROJECT IS UNFINISHED - ALMOST COMPLETE, MAIN TRADING LOOP IN-PROGRESS

This projects aims for automated trading on Bybit.com taking usage of their API. Project fetches live stock data in loop, takes trained LSTM (Long short term memory RNN) model and based on its prediction + some basic trading strategies (Golden cross etc.) makes signal to Buy or Sell. 
Project aims to backtest this strategy on historical data and make live trades on Bybit.com (yet to be finished). 
Project works with 5 minute freqency of price candles so data are quite large (300k rows) -> it's able to take advantage of LSTM predictions.
Project was also created to try to train Tensorflow models locally on GPU.


## Tools used
- Python (Pandas, Numpy, Tensorflow, Matplotlib)
- Bybit.com API