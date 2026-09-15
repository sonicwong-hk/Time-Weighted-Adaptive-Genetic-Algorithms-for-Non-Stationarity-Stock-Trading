import numpy as np

def sortino_ratio(portfolio_returns, risk_free_rate=0):
    expected_return = np.mean(portfolio_returns)
    

    negative_returns_sum = sum(x for x in portfolio_returns if x < 0)

    downside_returns = np.minimum(0, negative_returns_sum  - risk_free_rate)
    
    downside_deviation = np.sqrt(np.mean(downside_returns ** 2))
    
    if downside_deviation == 0:
        return 0
    
    sortino_ratio = (expected_return - risk_free_rate) / downside_deviation
    
    return sortino_ratio


def sharpe_ratio(portfolio_returns, risk_free_rate=0):
    expected_return = np.mean(portfolio_returns)
    volatility = np.std(portfolio_returns)
    
    if volatility == 0:
        return 0
    
    sharpe_ratio = (expected_return - risk_free_rate) / volatility
    
    return sharpe_ratio
