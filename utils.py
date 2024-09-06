import pandas as pd
import numpy as np
import sys
from pathlib import Path

def show_argv_path():
    print(f"le répertoire courant est {Path.cwd()}")
    print(f"le point d'entrée du programme est {sys.argv[0]}")
    print(f"la variable sys.path contient")
    for i, path in enumerate(sys.path, 1):
        print(f"{i}-ème chemin dans sys.path {path}")

def colonnes_nan(df, threshold=0):
    """enables retrieving columns thas have NANs"""
    for col in df.columns:
        if df[col].isnull().sum() > threshold:
            yield col, df[col].isnull().sum()

def compteur_nan(df, threshold=0):
    """enables retrieving indices of rows that have NANs"""
    for ind in df.index: #The index (row labels) of the DataFrame. (from documentation)
        nan_val = 0
        for value in df.loc[ind] :
            if pd.isna(value) == True:
                nan_val += 1
        if nan_val > threshold: #compter lignes avec NAN
            yield ind #on veut examiner nombres nan par index

def affecter_nan(df):
    """slow function replaces NAN with mean of the column"""
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            for ind, val in enumerate(df[col]): #ind ici est rang dans series
                if pd.isna(val) == True:
                    df.loc[df.index[ind], col] = df[col].mean()

def moyenne_agg(df):
    moyennes = [df[i].mean() for i in df.columns[:-1]]
    if len(moyennes)+1 == len(df.columns):
        nouvelle_ligne = moyennes+['maybe']
        s = pd.Series(dict(zip(df.columns, nouvelle_ligne)))
        return s

def var_interest(pca_object, pca_df, num_dims=3, threshold=0):
    """this functions iterates in the variables
    if that variables is well projected meaning long enough in the subspace defined by best PCs,
    then retrieve the name of variable from rank and vector norm in first four PCs"""
    for rank, i in enumerate(np.transpose(pca_object.components_)):
        if np.linalg.norm(i[:num_dims]) > threshold:
            yield pca_df.columns[rank], np.linalg.norm(i[:4])
