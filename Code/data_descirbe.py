import pandas as pd
import pandas_ta as ta
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

symbols = ['AAPL', 'DD', 'GS', 'JPM', 'MSFT',  'GE']
adj_close_data = {}
for symbol in symbols:
    filename = f'preprocessed_data/{symbol}_preprocessed_data.csv'
    data = pd.read_csv(filename, skipfooter=1, engine='python')
    print(f"\n{symbol} Stock Description:")
    result = data.describe()
    result.to_csv(f'{symbol}_statistics.csv')

