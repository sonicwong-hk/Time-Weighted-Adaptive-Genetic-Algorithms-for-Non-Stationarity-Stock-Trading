import pandas as pd
import neat
from neat import StatisticsReporter, StdOutReporter
from neat.nn import FeedForwardNetwork
from neat.config import Config
from neat.population import Population
from neat.genome import DefaultGenome
from neat.reproduction import DefaultReproduction
from neat.species import DefaultSpeciesSet
from neat.stagnation import DefaultStagnation
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
from helper import sortino_ratio, sharpe_ratio

random.seed(9001)
training_window_size = 120
testing_windows_size = 40

stocks = ['AAPL', 'MSFT', 'GS', 'JPM', 'DD', 'GE']

result_all = []

class TimeWeightedPopulation(neat.Population):
    def __init__(self, config, initial_state=None, time_factor=0.9):
        super().__init__(config, initial_state)
        self.time_factor = time_factor

    def reproduce(self, config, species, pop_size, generation):
        # Get the fitnesses and their respective weights
        fit_weights = [(i, s.fitness) for i, s in enumerate(species.members.values())]

        # Apply time weighting
        fit_weights = [(i, f * (self.time_factor ** i)) for i, f in fit_weights]

        # Normalize the weights
        total = sum(f for _, f in fit_weights)
        fit_weights = [(i, f / total) for i, f in fit_weights]

        # Select an individual based on the time-weighted probabilities
        chosen_index = random.choices([i for i, _ in fit_weights], weights=[f for _, f in fit_weights], k=1)[0]
        return species.members.values()[chosen_index]


for stock in stocks:

    print(stock)

    data = pd.read_csv(f'preprocessed_data/{stock}_preprocessed_data.csv')


    long_capital = 10000
    short_capital = 10000

    config = Config(DefaultGenome, DefaultReproduction,
                        DefaultSpeciesSet, DefaultStagnation,
                        'config_neat2.ini')

    config.selector = TimeWeightedPopulation(config, time_factor=0.9)


    config_short = Config(DefaultGenome, DefaultReproduction,
                        DefaultSpeciesSet, DefaultStagnation,
                        'config_neat_short2.ini')
    config_short.selector = TimeWeightedPopulation(config, time_factor=0.9)



    num_long = 0
    num_short = 0
    holding_long = False
    holding_short = False
    buy_price = 0
    long_win_num = 0
    short_win_num = 0

    trade_records = []

    for i in range(0, len(data) - training_window_size-testing_windows_size, testing_windows_size):
        train_data = data.iloc[i:i+training_window_size].copy()
        test_data = data.iloc[i+training_window_size+1:i+training_window_size+1+testing_windows_size].copy()

        train_data['daily_return_minmax'] = MinMaxScaler().fit_transform(train_data['daily_return'].values.reshape(-1, 1))
        test_data['daily_return_minmax'] = MinMaxScaler().fit_transform(test_data['daily_return'].values.reshape(-1, 1))

        def eval_genomes(genomes, config):
            for genome_id, genome in genomes:
                net = FeedForwardNetwork.create(genome, config)
                holding_stock = False
                buy_price = 0
                portfolio_returns = []

                for j in range(6, training_window_size):

                    thresholds = [int(x*100) for x in net.activate((np.array(train_data['daily_return_minmax'].iloc[j-5:j])*100))]
                    
                    if  not (thresholds[0] == 0 and thresholds[1] == 0):

                        rsi_buy = 50*thresholds[0] / (thresholds[0] + thresholds[1])
                        rsi_sell = 100*thresholds[1] / (thresholds[0] + thresholds[1])

                        buy_signal = (train_data['rsi'].iloc[j] < rsi_buy) * 1
                        sell_signal = (train_data['rsi'].iloc[j] > rsi_sell) * -1

                        if buy_signal == 1 and not holding_stock:
                            if train_data['adjClose'].iloc[j] > train_data['ema'].iloc[j]:
                                buy_price = train_data['adjClose'].iloc[j]
                                holding_stock = True
                        elif train_data['adjClose'].iloc[j] < train_data['ema'].iloc[j] and sell_signal == -1 and holding_stock:
                            sell_price = train_data['adjClose'].iloc[j]
                            portfolio_returns.append(sell_price / buy_price - 1)
                            holding_stock = False

                if holding_stock == True:
                    sell_price = train_data['adjClose'].iloc[-1]
                    profit = sell_price / buy_price - 1
                    portfolio_returns.append(profit)

                genome.fitness = sharpe_ratio(portfolio_returns)

        def eval_genomes_short(genomes, config):
            for genome_id, genome in genomes:
                net = FeedForwardNetwork.create(genome, config)
                holding_stock = False
                sell_price = 0
                portfolio_returns = []

                for j in range(6, training_window_size):
                    thresholds = [int(x*100) for x in net.activate((np.array(train_data['daily_return_minmax'].iloc[j-5:j])*100))]
                    if  not (thresholds[0] == 0 and thresholds[1] == 0):

                        rsi_buy = 50*thresholds[0] / (thresholds[0] + thresholds[1])
                        rsi_sell = 100*thresholds[1] / (thresholds[0] + thresholds[1])

                        buy_signal = (train_data['rsi'].iloc[j] < rsi_buy) * 1
                        sell_signal = (train_data['rsi'].iloc[j] > rsi_sell) * -1

                        if sell_signal == -1 and not holding_stock:
                            if train_data['adjClose'].iloc[j] < train_data['ema'].iloc[j]:
                                sell_price = train_data['adjClose'].iloc[j]
                                holding_stock = True
                        elif train_data['adjClose'].iloc[j] > train_data['ema'].iloc[j] and buy_signal == 1 and holding_stock:
                            buy_price = train_data['adjClose'].iloc[j]
                            portfolio_returns.append(sell_price / buy_price - 1)
                            holding_stock = False
                if holding_stock == True:
                    buy_price = train_data['adjClose'].iloc[-1]
                    portfolio_returns.append(sell_price / buy_price - 1)

                genome.fitness = sharpe_ratio(portfolio_returns)
    
        p_long = Population(config)
        p_short = Population(config_short)
        stats = StatisticsReporter()
        p_long.add_reporter(StdOutReporter(False))
        p_long.add_reporter(stats)
        p_short.add_reporter(StdOutReporter(False))
        p_short.add_reporter(stats)

        winner_long = p_long.run(eval_genomes, 5)
        winner_long_net = FeedForwardNetwork.create(winner_long, config)

        winner_short = p_short.run(eval_genomes_short, 5)
        winner_short_net = FeedForwardNetwork.create(winner_short, config)

        holding_long = False
        holding_short = False

        long_buy_date = "na"
        long_sell_date = "na"
        short_buy_date = "na"
        short_sell_date = "na"

        for j in range(6, testing_windows_size):
            optimal_thresholds_long = [int(x*100) for x in winner_long_net.activate((np.array(test_data['daily_return_minmax'].iloc[j-5:j])*100))]
            optimal_thresholds_short = [int(x*100) for x in winner_short_net.activate((np.array(test_data['daily_return_minmax'].iloc[j-5:j])*100))]

            rsi_buy_long = 50*optimal_thresholds_long[0] / (optimal_thresholds_long[0] + optimal_thresholds_long[1])
            rsi_sell_long = 100*optimal_thresholds_long[1] / (optimal_thresholds_long[0] + optimal_thresholds_long[1])

            rsi_buy_short = 50*optimal_thresholds_long[0] / (optimal_thresholds_short[0] + optimal_thresholds_short[1])
            rsi_sell_short = 100*optimal_thresholds_short[1] / (optimal_thresholds_short[0] + optimal_thresholds_short[1])

            buy_signal_long = (train_data['rsi'].iloc[j] < rsi_buy_long) * 1
            sell_signal_long = (train_data['rsi'].iloc[j] > rsi_sell_long) * -1

            buy_signal_short = (train_data['rsi'].iloc[j] < rsi_buy_short ) * 1
            sell_signal_short = (train_data['rsi'].iloc[j] > rsi_sell_short) * -1

            if buy_signal_long == 1:
                if (test_data['adjClose'].iloc[j] > test_data['ema'].iloc[j]) and not holding_long:
                    buy_price_long = test_data.iloc[j]['adjClose']
                    holding_long = True
                    long_buy_date = test_data.iloc[j]['date']
            elif test_data['adjClose'].iloc[j] < test_data['ema'].iloc[j] and sell_signal_long == -1 and holding_long:
                sell_price = test_data.iloc[j]['adjClose']
                long_capital *= sell_price / buy_price_long
                long_sell_date = test_data.iloc[j]['date']
                trade_records.append(['Long',  buy_price_long, sell_price, sell_price /  buy_price_long, long_capital, long_buy_date, long_sell_date])
                if (sell_price / buy_price_long) > 1:
                    long_win_num += 1
                num_long += 1
                print('Long', buy_price_long, sell_price, sell_price / buy_price_long, long_capital)
                holding_long = False


            if sell_signal_short == -1:
                if (test_data['adjClose'].iloc[j] < test_data['ema'].iloc[j]) and not holding_short:
                    sell_price_short = test_data.iloc[j]['adjClose']
                    holding_short = True
                    short_buy_date = test_data.iloc[j]['date']
            elif test_data['adjClose'].iloc[j] > test_data['ema'].iloc[j] and buy_signal_short == 1 and holding_short:
                buy_price = test_data.iloc[j]['adjClose']
                short_capital *= sell_price_short / buy_price
                short_sell_date = test_data.iloc[j]['date']
                trade_records.append(['Short', sell_price_short , buy_price, sell_price_short  / buy_price, short_capital, short_buy_date, short_sell_date])

                if (sell_price_short  / buy_price)> 1:
                    short_win_num += 1
                num_short += 1
                print('Short', sell_price_short , buy_price, sell_price_short  / buy_price, short_capital)
                holding_short = False

    print(long_capital, short_capital)

    columns = ['Type', 'Entry Price', 'Exit Price', 'Profit Factor', 'Capital', 'Open Position Date', 'Close Position Date']
    trade_df = pd.DataFrame(trade_records, columns=columns)
    trade_df.to_csv(f'trade_records_{stock}2.csv', index=False)
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
    
    result_all.append([stock, (long_return-1)*100, num_long, long_win_rate, long_capital, (short_return-1)*100, num_short, short_win_rate, short_capital, total_return*100])

columns_all = ['Stock', 'long return(%)', 'Num of Long', 'Long win rate(%)', 'Final Capital of Long', 'short return(%)', 'Num of Short', 'Short win rate(%)', 'Final Capital of Short', 'total return(%)']
result_df = pd.DataFrame(result_all, columns=columns_all)
result_df.to_csv(f'stock_performance2.csv', index=False)