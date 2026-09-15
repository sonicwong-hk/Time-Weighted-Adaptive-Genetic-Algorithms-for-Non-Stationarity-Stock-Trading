import pandas as pd
import matplotlib.pyplot as plt

stocks = ['AAPL', 'MSFT', 'GS', 'JPM', 'DD', 'GE']

for stock in stocks:
    data = pd.read_csv(f'preprocessed_data/{stock}_preprocessed_data.csv')
    trading_record =  pd.read_csv(f'balance/balance_{stock}.csv')
    trading_record_original =  pd.read_csv(f'balance/balance_{stock}_original.csv')

    # trading_record =  pd.read_csv(f'result (Time-weighted Adaptive NEAT)/trade_records_{stock}.csv')
    # trading_record_original =  pd.read_csv(f'result (Original NEAT)/trade_records_{stock}.csv')
    # print(stock)

    # starting_balance = 20000
    # long_balance = 10000
    # short_balance = 10000

    # close_position_dates = []
    # total_returns = []

    # for index, row in trading_record.iterrows():
    #     type_ = row['Type']

    #     capital = row['Capital']
    #     if type_ == 'Long':
    #         long_balance = capital
    #     elif type_ == 'Short':
    #         short_balance = capital

    #     close_position_date = row['Close Position Date']
    #     total_balance = (long_balance+short_balance)
    #     total_return = total_balance/starting_balance

    #     close_position_dates.append(close_position_date)
    #     total_returns.append(total_return)
    
    # # Fill missing dates with previous row's value
    # df = pd.DataFrame({'close_position_date': close_position_dates, 'total_return': total_returns})  
    # full_dates = pd.DataFrame({'date': data['date']})
    # merged_df = pd.merge(full_dates, df, left_on='date', right_on='close_position_date', how='left')
    # merged_df['total_return'] = merged_df['total_return'].fillna(method='ffill')

    # # Save the updated DataFrame
    # merged_df.to_csv(f'balance_{stock}.csv', index=False)

    data['date'] = pd.to_datetime(data['date'])
    trading_record['date'] = pd.to_datetime(trading_record['date'])
    trading_record_original['date'] = pd.to_datetime(trading_record_original['date'])

    # Divide all values in the adjClose column by the first value
    first_adjClose = data['adjClose'].iloc[0]
    data['adjClose'] = data['adjClose'] / first_adjClose

    # Merge the two DataFrames on the date column
    merged_df = pd.merge(data, trading_record, left_on='date', right_on='date', how='inner')
    merged_df_original = pd.merge(data, trading_record_original, left_on='date', right_on='date', how='inner')

    # Plot the graph
    plt.figure(figsize=(10, 6))
    plt.plot(merged_df['date'], merged_df['adjClose'], label='Price')
    plt.plot(merged_df['date'], merged_df['total_return'], label='Total Return (Time-weighted NEAT)')
    plt.plot(merged_df_original['date'], merged_df_original['total_return'], label='Total Return (Original NEAT)')
    plt.title(f'{stock} - Stock Price vs Total Return of Portfolio')
    plt.xlabel('Year')
    plt.ylabel('Return (In log scale)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.yscale('log') 
    plt.tight_layout()
    plt.savefig(f'total_return_{stock}')


    # Calculate volatility
    volatility_adjClose = data['adjClose'].pct_change().rolling(window=10).std()
    volatility_total_return = merged_df['total_return'].pct_change().rolling(window=10).std()
    volatility_total_return_original = merged_df_original['total_return'].pct_change().rolling(window=10).std()


    # Calculate the difference in volatility
    volatility_diff_neat = volatility_total_return - volatility_adjClose
    volatility_diff_original = volatility_total_return_original - volatility_adjClose

    # Plot the difference in volatility
    plt.figure(figsize=(10, 6))
    plt.plot(data['date'], volatility_diff_original*100, label='Difference in Volatility (Original NEAT)')
    plt.plot(data['date'], volatility_diff_neat*100, label='Difference in Volatility (Time-weighted NEAT)')
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1) 
    plt.title(f'{stock} - Difference in Volatility Comparison')
    plt.xlabel('Year')
    plt.ylabel('Difference in Volatility (In %)')
    plt.legend()
    plt.text(data['date'].iloc[0], -6, 'Above 0: More volatile than stock Price\nBelow 0: Less Volatile than stock Price', fontsize=12, color='green')

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'volatity_{stock}')
