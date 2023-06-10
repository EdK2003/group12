import pandas as pd
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

df_data_raw = pd.read_csv('C:\\Git files\\group12\\mol_desc_given.csv', index_col='SMILES')
header = df_data_raw.columns
index = df_data_raw.index
df_data_scaled = pd.DataFrame(scaler.fit_transform(df_data_raw), index=index, columns=header)

df_corr = df_data_scaled.loc[:,'MaxAbsEStateIndex':'fr_urea'].corr()

print()