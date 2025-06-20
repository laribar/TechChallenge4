import pandas as pd
import os
from functools import reduce

def load_nhanes_data(path="nhanes_data"):
    """Carrega e combina arquivos NHANES por SEQN."""

    files = [
        "DEMO_L.xpt",
        "BMX_L.xpt",
        "DR1TOT_L.xpt",
        "DIQ_L.xpt",
        "MCQ_L.xpt",
        "DPQ_L.xpt",
        "PAQ_L.xpt",
        "SLQ_L.xpt",
        "SMQ_L.xpt",
        "DBQ_L.xpt",
        "ALQ_L.xpt",
        "WHQ_L.xpt",
        "OCQ_L.xpt"
    ]

    dfs = []
    for file in files:
        file_path = os.path.join(path, file)
        print(f"📥 Carregando {file_path}...")

        df_temp = pd.read_sas(file_path)
        print(f"➡️  {file} tem {df_temp.shape[0]} linhas e {df_temp.shape[1]} colunas.")

        dfs.append(df_temp)

    df_final = reduce(lambda left, right: pd.merge(left, right, on='SEQN', how='left'), dfs)

    print(f"\n✅ Dados combinados com sucesso: {df_final.shape[0]} linhas, {df_final.shape[1]} colunas.")
    return df_final
