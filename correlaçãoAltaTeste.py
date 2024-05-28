import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Generate example data
np.random.seed(0)
x = np.random.rand(100)
y = 2 * x + np.random.normal(0, 0.1, 100)  # y is highly correlated with x

# Create a DataFrame
df = pd.DataFrame({'x': x, 'y': y})

# Calculate correlation
correlation = df.corr().iloc[0, 1]
print(f'Correlation between x and y: {correlation}')

# Plotting the scatter plot
plt.scatter(df['x'], df['y'], marker='+', color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter plot of highly correlated variables (x vs y)')
plt.show()