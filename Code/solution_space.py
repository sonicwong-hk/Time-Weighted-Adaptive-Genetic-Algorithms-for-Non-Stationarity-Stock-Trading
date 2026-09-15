import pandas as pd
import numpy as np
import neat
import matplotlib.pyplot as plt
from neat.nn import FeedForwardNetwork

def evaluate_genome(genomes, config):
    data = pd.read_csv(f'preprocessed_data/AAPL_preprocessed_data.csv')
    max_fitness = 0
    
    for genome_id, genome in genomes:
        net = FeedForwardNetwork.create(genome, config)
        holding_stock = False
        buy_price = 0
        portfolio_returns = []

        for j in range(6, len(data)):

            thresholds = [int(x*100) for x in net.activate((np.array(data['adjClose'].iloc[j-5:j])))]
            
            if  not (thresholds[0] == 0 and thresholds[1] == 0):

                rsi_buy = 50*thresholds[0] / (thresholds[0] + thresholds[1])
                rsi_sell = 100*thresholds[1] / (thresholds[0] + thresholds[1])

                buy_signal = (data['rsi'].iloc[j] < rsi_buy) * 1
                sell_signal = (data['rsi'].iloc[j] > rsi_sell ) * -1

                if buy_signal == 1 and not holding_stock:
                    if data['adjClose'].iloc[j] > data['ema'].iloc[j]*0.95:
                        buy_price = data['adjClose'].iloc[j]
                        holding_stock = True
                elif sell_signal == -1 and holding_stock:
                    sell_price = data['adjClose'].iloc[j]
                    portfolio_returns.append(sell_price / buy_price - 1)
                    holding_stock = False

        if holding_stock == True:
            sell_price = data['adjClose'].iloc[-1]
            profit = sell_price / buy_price - 1
            portfolio_returns.append(profit)

        profit = np.mean(portfolio_returns)
        if profit > max_fitness:
            max_fitness = profit
        genome.fitness = profit
    
    return max_fitness  # Return the maximum fitness value

def run():
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'config_neat.ini')

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Store the maximum fitness value of each generation
    max_fitness_values = []
    for i in range(5):  # Run for 5 generations
        max_fitness = p.run(evaluate_genome, 1)  # Run for 1 generation at a time
        max_fitness_values.append(max_fitness)  # Store the max fitness for this generation

    # Plot the maximum fitness value of each generation
    plt.plot(range(1, 6), max_fitness_values)
    plt.xlabel('Generation')
    plt.ylabel('Max Fitness Value')
    plt.title('NEAT Max Fitness Value over Generations')
    plt.show()

run()
