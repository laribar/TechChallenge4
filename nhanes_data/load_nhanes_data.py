import pandas as pd
import os
from functools import reduce

def load_nhanes_data(path="nhanes_data"):
    """Carrega e combina arquivos NHANES por SEQN."""
    files = [
        "DEMO_L.XPT",
        "BMX_L.XPT",
        "DR1TOT_L.XPT",
        "DIQ_L.XPT",
        "MCQ_L.XPT",
        "DPQ_L.XPT",
        "PAQ_L.XPT",
        "SLQ_L.XPT",
        "SMQ_L.XPT",
        "DBQ_L.XPT",
        "ALQ_L.XPT",
        "WHQ_L.XPT",
        "OCQ_L.XPT"
    ]

    dfs = []
    for file in files:
        file_path = os.path.join(path, file)
        print(f"Carregando {file_path}...")
        dfs.append(pd.read_sas(file_path))

    df_final = reduce(lambda left, right: pd.merge(left, right, on='SEQN', how='inner'), dfs)

    print(f"✅ Dados combinados com sucesso: {df_final.shape[0]} linhas, {df_final.shape[1]} colunas.")
    return df_final

