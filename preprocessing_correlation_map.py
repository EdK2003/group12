import pandas as pd
import numpy as np
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import PandasTools
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

def calculate_mol_descriptors(df):
    calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
    header = calc.GetDescriptorNames()
    rdkit_2d_desc = []
    mols = df['ROMol'].tolist()
    for i in range(len(df)):
        ds = calc.CalcDescriptors(mols[i])
        rdkit_2d_desc.append(ds)
    df_out = pd.DataFrame(index=df['SMILES'], columns=header, data=rdkit_2d_desc)
    return df_out

# Read data from files
df1 = pd.read_csv('tested_molecules_v2.csv')
df2 = pd.read_csv('tested_molecules-1.csv')

# Combine dataframes
raw_mols = pd.concat((df1, df2))

# Calculate descriptors
PandasTools.AddMoleculeColumnToFrame(raw_mols, smilesCol='SMILES')
df_desc = calculate_mol_descriptors(raw_mols)
df_desc['ALDH1_inhibition'] = raw_mols['ALDH1_inhibition'].values.copy()

# Scale data
scaler = MinMaxScaler()
header = df_desc.columns
index = df_desc.index
df_data_scaled = pd.DataFrame(scaler.fit_transform(df_desc), index=index, columns=header)

# Compute correlation matrix
df_corr = df_data_scaled.loc[:, 'MaxAbsEStateIndex':'fr_urea'].corr()

# Drop highly correlated variables
upper_tri = df_corr.where(np.triu(np.ones(df_corr.shape), k=1).astype(bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
df_data_scaled.drop(to_drop, axis=1, inplace=True)

# Drop columns with missing data
missing_data = df_data_scaled.isnull()
for col in missing_data:
    if True in missing_data[col].unique():
        df_data_scaled.drop(col, axis=1, inplace=True)

# Drop duplicate rows
df_data_scaled.drop_duplicates(inplace=True)

# Plot correlation coefficients map
plt.figure(figsize=(12, 10))
sns.heatmap(df_corr, cmap='coolwarm', annot=True, fmt=".2f", linewidths=0.5)
plt.title('Correlation Coefficients Map')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()





