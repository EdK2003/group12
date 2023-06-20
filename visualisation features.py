import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV files
names_data = pd.read_csv('top100.csv')
values_data = pd.read_csv('top100_details.csv')

# Select the desired features
selected_features = ['MolWt', 'MolLogP', 'TPSA', 'NumHAcceptors', 'NumHDonors', 'NumRotatableBonds', 'FractionCSP3']

# Create an empty dictionary to store the feature values
feature_values = {}

# Link rows based on row index
for index, row in names_data.iterrows():
    molecule_name = row[0]
    feature_values[molecule_name] = values_data.iloc[index].loc[selected_features]

# Create separate plots for each feature
for feature in selected_features:
    # Get the feature values
    values = [feature_values[molecule][feature] for molecule in names_data.iloc[:, 0]]

    # Plot the values
    plt.figure(figsize=(10, 6))
    plt.plot(values, marker='o')
    plt.xlabel('Molecule')
    plt.ylabel(feature)
    plt.title(f'Plot of {feature} for Molecules')
    plt.xticks(rotation=90)
    plt.show()
