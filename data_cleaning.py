import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# read data from file
df_data_raw = pd.read_csv('C:\\Git files\\group12\\mol_desc_given.csv', index_col='SMILES')

# scale data
scaler = MinMaxScaler()
header = df_data_raw.columns
index = df_data_raw.index
df_data_scaled = pd.DataFrame(scaler.fit_transform(df_data_raw), index=index, columns=header)

df_corr = df_data_scaled.loc[:,'MaxAbsEStateIndex':'fr_urea'].corr()

# drop highly correlated variables
upper_tri = df_corr.where(np.triu(np.ones(df_corr.shape),k=1).astype(bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
df_data_scaled.drop(to_drop, axis=1, inplace=True)

# drop columns with missing data
missing_data = df_data_scaled.isnull()
for col in missing_data:
    if True in missing_data[col].unique():
        df_data_scaled.drop(col, axis=1, inplace=True)
        
# rop duplicate rows
df_data_scaled.drop_duplicates(inplace=True)
