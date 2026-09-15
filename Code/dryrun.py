import pandas as pd

stocks = ['AAPL', 'MSFT', 'GS', 'JPM', 'DD', 'GE']
training_window_size = 120
testing_windows_size = 40
result_all = []

# para = {
#     'AAPL': [55, 75, 65, 85],
#     'MSFT': [50, 70, 60, 80],
#     'GS': [60, 85, 84, 85],
#     'JPM': [50, 80, 65, 85],
#     'DD': [40, 75, 40, 80],
#     'GE': [30, 80, 40, 80]
# }

para = {
    'AAPL': [40, 65, 45, 80],
    'MSFT': [30, 50, 45, 70],
    'GS': [40, 85, 55, 85],
    'JPM': [40, 70, 45, 85],
    'DD': [40, 65, 50, 80],
    'GE': [40, 80, 50, 80]
}

for stock in stocks:
    data = pd.read_csv(f'preprocessed_data/{stock}_preprocessed_data.csv', skiprows=[1])
    print(stock)

    long_capital = 10000
    short_capital = 10000

    long_buy_date = "na"
    long_sell_date = "na"
    short_buy_date = "na"
    short_sell_date = "na"

    num_long = 0
    num_short = 0
    holding_long = False
    holding_short = False
    buy_price = 0
    long_win_num = 0
    short_win_num = 0

    trade_records = []
    for j in range(0, len(data) - training_window_size-testing_windows_size):
        
        buy_signal_long = (data['rsi'].iloc[j] < para[stock][0]) * 1
        sell_signal_long = (data['rsi'].iloc[j] > para[stock][1]) * -1

        buy_signal_short = (data['rsi'].iloc[j] < para[stock][2]) * 1
        sell_signal_short = (data['rsi'].iloc[j] > para[stock][3]) * -1

        if buy_signal_long == 1:
            if (data['adjClose'].iloc[j] > data['ema'].iloc[j]*0.95) and not holding_long:
                buy_price_long = data.iloc[j]['adjClose']
                holding_long = True
                long_buy_date = data.iloc[j]['date']
        elif sell_signal_long == -1 and holding_long:
            sell_price = data.iloc[j]['adjClose']
            long_capital *= sell_price / buy_price_long
            long_sell_date = data.iloc[j]['date']
            trade_records.append(['Long',  buy_price_long, sell_price, sell_price /  buy_price_long, long_capital, long_buy_date, long_sell_date])
            if (sell_price / buy_price_long) > 1:
                long_win_num += 1
            num_long += 1
            holding_long = False

        if sell_signal_short == -1:
            if (data['adjClose'].iloc[j]*0.98 < data['ema'].iloc[j]) and not holding_short:
                sell_price_short = data.iloc[j]['adjClose']
                holding_short = True
                short_buy_date = data.iloc[j]['date']
        elif buy_signal_short == 1 and holding_short:
            buy_price = data.iloc[j]['adjClose']
            short_capital *= sell_price_short / buy_price
            short_sell_date = data.iloc[j]['date']
            trade_records.append(['Short', sell_price_short , buy_price, sell_price_short  / buy_price, short_capital, short_buy_date, short_sell_date])

            if (sell_price_short  / buy_price)> 1:
                short_win_num += 1
            num_short += 1
            holding_short = False

    columns = ['Type', 'Entry Price', 'Exit Price', 'Profit Factor', 'Capital', 'Open Position Date', 'Close Position Date']
    trade_df = pd.DataFrame(trade_records, columns=columns)
    trade_df.to_csv(f'trade_records_{stock}.csv', index=False)
    long_return = long_capital/10000
    short_return = short_capital/10000
    total_return = (((long_capital+short_capital)/20000)-1)

    if num_long == 0:
        long_win_rate = 0
    else:
        long_win_rate = long_win_num*100/num_long
    
    if num_short == 0:
        short_win_rate =0
    else:
        short_win_rate = short_win_num*100/num_short
    
    annual_long = ((long_capital/10000)**(1/10)-1)*100
    annual_short = ((short_capital/10000)**(1/10)-1)*100

    result_all.append([stock, (long_return-1)*100, num_long, long_win_rate, long_capital, (short_return-1)*100, num_short, short_win_rate, short_capital, total_return*100, annual_long, annual_short])
    
columns_all = ['Stock', 'long return(%)', 'Num of Long', 'Long win rate(%)', 'Final Capital of Long', 'short return(%)', 'Num of Short', 'Short win rate(%)', 'Final Capital of Short', 'total return(%)','annaulized return of Long', 'annaulized return of Short']
result_df = pd.DataFrame(result_all, columns=columns_all)
result_df.to_csv(f'stock_performance.csv', index=False)

print(result_df)