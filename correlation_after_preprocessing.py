import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Read preprocessed data from file
df_data_scaled = pd.read_csv('mol_desc_given.csv', index_col=0)

# Compute correlation matrix
df_corr = df_data_scaled.corr()

# Plot correlation coefficients map
plt.figure(figsize=(12, 10))
sns.heatmap(df_corr, cmap='coolwarm', annot=True, fmt=".2f", linewidths=0.5)
plt.title('Correlation Coefficients Map (Preprocessed Data)')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()
