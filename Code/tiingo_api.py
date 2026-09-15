import pandas as pd
from tiingo import TiingoClient

config = {'session': True, 'api_key': 'd35cafc284131bf40deb9ce194b77f6f22884cf5'}
client = TiingoClient(config)

symbols = ['AAPL', 'AXP', 'BA', 'CAT', 'CSCO', 'CVX', 'DIS', 'DD', 'GS', 'HD',
           'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE',
           'PG', 'TRV', 'UNH', 'RTX', 'VZ', 'WBA', 'WMT', 'XOM', 'GE', 'PFE']

all_data = pd.DataFrame()

for symbol in symbols:
    data = client.get_dataframe(symbol,
                                frequency='daily',
                                metric_name='adjClose',
                                startDate='2013-01-01',
                                endDate='2022-12-31')
    
    data['symbol'] = symbol
    data['date'] = data.index
    filename = f'stockdata/{symbol}_stock_data.csv'

    data.to_csv(filename)