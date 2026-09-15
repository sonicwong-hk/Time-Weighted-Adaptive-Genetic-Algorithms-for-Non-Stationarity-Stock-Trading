import pandas as pd
import pandas_ta as ta
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

symbols = ['AAPL', 'AXP', 'BA', 'CAT', 'CSCO', 'CVX', 'DIS', 'DD', 'GS', 'HD',
           'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE',
           'PG', 'TRV', 'UNH', 'RTX', 'VZ', 'WBA', 'WMT', 'XOM', 'GE', 'PFE']
adj_close_data = {}
for symbol in symbols:
    filename = f'stockdata/{symbol}_stock_data.csv'
    data = pd.read_csv(filename, skipfooter=1, engine='python')
    data['adjClose'].fillna(method='ffill', inplace=True)
    data['adjClose'] = pd.to_numeric(data['adjClose'], errors='coerce')

    data['daily_return'] = data['adjClose'].pct_change() * 100
    data['daily_return'].fillna(0, inplace=True)
    data['ema'] = ta.ema(data['adjClose'], length=3)
    data['ema_daily_return'] = ta.ema(data['daily_return'], length=3)
    data['rsi'] = ta.rsi(data['adjClose'], length=3)
    data['rsi_daily_return'] = ta.rsi(data['daily_return'], length=3)


    data = data.iloc[:-1]
    output_filename = f'preprocessed_data/{symbol}_preprocessed_data.csv'
    data.to_csv(output_filename, index=False)
    adj_close_data[symbol] = data['adjClose']


correlation_matrix = pd.DataFrame({symbol: series for symbol, series in adj_close_data.items()}).corr()

aapl_corr = correlation_matrix['AAPL'].sort_values(ascending=False)


# Plotting the graph
plt.figure(figsize=(10, 6))
sns.barplot(x=aapl_corr.index, y=aapl_corr.values, palette='viridis')
plt.title('Correlation of AAPL with Other Stocks')
plt.xlabel('Stock Symbols')
plt.ylabel('Correlation')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
