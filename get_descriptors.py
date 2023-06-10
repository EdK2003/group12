import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import PandasTools


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


df1 = pd.read_csv('C:\\Git files\\group12\\tested_molecules_v2.csv')
df2 = pd.read_csv('C:\\Git files\\group12\\tested_molecules-1.csv')

raw_mols = pd.concat((df1, df2))

PandasTools.AddMoleculeColumnToFrame(raw_mols, smilesCol='SMILES')

df_desc = calculate_mol_descriptors(raw_mols)\

df_desc['ALDH1_inhibition'] = raw_mols['ALDH1_inhibition'].values.copy()

df_desc.to_csv('C:\\Git files\\group12\\mol_desc_given.csv')








