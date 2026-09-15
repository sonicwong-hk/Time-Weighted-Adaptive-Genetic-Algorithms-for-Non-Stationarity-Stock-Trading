import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV file
data = pd.read_csv('trade_records_MSFT2.csv')

# Filter for Short trades and convert 'Close Position Date' to datetime
data = data[data['Type'] == 'Short']
data['Close Position Date'] = pd.to_datetime(data['Close Position Date'])

# Sort data by 'Close Position Date'
data.sort_values('Close Position Date', inplace=True)

# Calculate the returns and cumulative capital
data['Returns'] = data['Capital'].pct_change()  # Returns between each point
data['Cumulative Returns'] = (1 + data['Returns']).cumprod()  # Cumulative product of (1 + Returns)

# Plotting
plt.figure(figsize=(10, 5))  # Set the figure size
plt.plot(data['Close Position Date'], data['Cumulative Returns'], marker='o', linestyle='-', color='b', markersize=5, label='Closing Position')  # Line plot with markers

plt.title('Cumulative Return Over Time for MSFT (Short Strategy)')  # Title of the plot
plt.xlabel('Close Position Date')  # X-axis label
plt.ylabel('Cumulative Return')  # Y-axis label
plt.grid(True)  # Add grid for better readability
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
plt.legend()  # Show legend
plt.tight_layout()  # Automatically adjust subplot parameters to give specified padding
plt.savefig('cumulative_capital.png')  # Save the plot

