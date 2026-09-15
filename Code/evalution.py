import pandas as pd
import numpy as np
stocks = ['AAPL', 'MSFT', 'GS', 'JPM', 'DD', 'GE']
for stock in stocks:
    print(stock)
    # Load your data
    data = pd.read_csv(f'result (Time-weighted Adaptive NEAT)/trade_records_{stock}.csv')

    # Filter records where Type is 'Long'
    long_type_data = data[data['Type'] == 'Long']

    # Assuming 'Profit Factor' is the daily returns from the filtered data
    daily_returns = long_type_data['Profit Factor']

    # Calculate the target (you can set this to mean, zero, or another benchmark)
    target = 1  # This can be changed to daily_returns.mean() if you prefer the mean

    # Filter for returns below the target
    negative_returns = daily_returns[daily_returns < target]

    # Calculate daily downside volatility
    downside_volatility = negative_returns.std()

    # Annualize the downside volatility
    annualized_downside_volatility = downside_volatility * np.sqrt(252)

    # Print the annualized downside volatility as a percentage
    print("Annualized Downside Volatility (%):", annualized_downside_volatility * 100)

    # Get the number of records
    number_of_records = long_type_data.shape[0]

    # Print the number of records
    print("Number of records:", number_of_records)

    # Calculate the number of records where Profit Factor > 1
    profitable_trades = long_type_data[long_type_data['Profit Factor'] > 1].shape[0]

    # Print the number of profitable trades
    print("Number of profitable trades (Profit Factor > 1):", profitable_trades)

    # Calculate and print the win rate
    win_rate = profitable_trades * 100 / number_of_records
    print("Win Rate (%):", win_rate)

    # Calculate and print the average return per trade
    average_return_per_trade = long_type_data['Profit Factor'].mean()
    print("Average Return per Trade:", (average_return_per_trade-1)*100)
