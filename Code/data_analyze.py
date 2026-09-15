import pandas as pd
from scipy.stats import pearsonr
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
from scipy.stats import ttest_ind

stocks = ['AAPL']
windows_index = 20
testing_windows = 40
for stock in stocks:
    print(stock)
    data = pd.read_csv(f'preprocessed_data/{stock}_preprocessed_data.csv')

    # plt.plot(data.index[windows_index+windows_index:], data['adjClose'][windows_index+windows_index:])
    # plt.xlabel('Index')
    # plt.ylabel('Close Price')
    # plt.title(f'Close Price plot of {stock}')
    # plt.grid(True)
    # plt.show()

    unseen_data = data['daily_return'].shift(-windows_index-windows_index).rolling(window=testing_windows).mean().dropna().reset_index(drop=True)
    more_recent_data = data['daily_return'].shift(-windows_index).rolling(window=testing_windows).mean().dropna().reset_index(drop=True).iloc[:-windows_index]
    less_recent_data = data['daily_return'].rolling(window=testing_windows).mean().dropna().reset_index(drop=True).iloc[:-windows_index-windows_index]

    # Calculate the correlation and the p-value
    corr_unseen_recent, p_value_unseen_recent =  pearsonr(unseen_data, more_recent_data)
    corr_unseen_less_recent, p_value_unseen_less_recent = pearsonr(unseen_data, less_recent_data)

    if corr_unseen_recent > corr_unseen_less_recent:
        print("more recent data is more correlated then less recent data")
    else:
        print("more recent data is less correlated then less recent data")

    print(f'Correlation between unseen data and more recent data: {corr_unseen_recent}')
    print(f'P-value between unseen data and more recent data: {p_value_unseen_recent}')

    print(f'Correlation between unseen data and less recent data: {corr_unseen_less_recent}')
    print(f'P-value between unseen data and less recent data: {p_value_unseen_less_recent}')

    if p_value_unseen_recent < 0.05:
        print("Reject the null hypothesis. The correlation between unseen data and more recent data is statistically significant.")
    else:
        print("Fail to reject the null hypothesis. The correlation between unseen data and more recent data is not statistically significant.")

    if p_value_unseen_less_recent < 0.05:
        print("Reject the null hypothesis. The correlation between unseen data and less recent data is statistically significant.")
    else:
        print("Fail to reject the null hypothesis. The correlation between unseen data and less recent data is not statistically significant.")

    # Calculate Fisher's Z transformation of correlation coefficients
    z_recent = 0.5 * np.log((1 + corr_unseen_recent) / (1 - corr_unseen_recent))
    z_less_recent = 0.5 * np.log((1 + corr_unseen_less_recent) / (1 - corr_unseen_less_recent))

    se_diff = np.sqrt(1 / (len(unseen_data) - 3) + 1 / (len(unseen_data) - 3))

    test_statistic = (z_recent - z_less_recent) / se_diff

    degrees_of_freedom = 2 * len(unseen_data) - 6
    p_value = 2 * t.cdf(-np.abs(test_statistic), df=degrees_of_freedom)

    print(f'The P-value of the t-test: {p_value}')

    if p_value < 0.05:
        print("Reject the null hypothesis. There is a significant difference between the two correlations.")
    else:
        print("Fail to reject the null hypothesis. There is no significant difference between the two correlations.")
