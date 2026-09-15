import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

stocks = ['AAPL', 'MSFT', 'GS', 'JPM', 'DD', 'GE']

def test_random_walk(data):
    result = adfuller(data)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    if result[1] > 0.05:
        print("The data has a unit root and appears to be a random walk.")
    else:
        print("The data does not have a unit root and does not appear to be a random walk.")

for stock in stocks:
    data = pd.read_csv(f'preprocessed_data/{stock}_preprocessed_data.csv', skiprows=[1])
    print(stock)

    # Test if the data is a random walk
    print("Performing ADF test:")
    test_random_walk(data['adjClose'].dropna())

    # Plot autocorrelation graph with more lags
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_acf(data['adjClose'].dropna(), ax=ax, lags=80)  # Specify the number of lags
    plt.title(f"Autocorrelation for {stock}")
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.grid(True)
    plt.savefig(f'ACF/{stock} acf.png')
