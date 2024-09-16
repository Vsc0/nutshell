import os
import pandas as pd
from pathlib import Path


def pd2csv(df: pd.DataFrame, dirname: str, filename: str) -> str:
    Path(dirname).mkdir(parents=True, exist_ok=True)
    path_or_buf = os.path.join(dirname, f'{filename}.csv')
    df.to_csv(path_or_buf=path_or_buf, index=False)
    return path_or_buf


def csv2pd(filename: str) -> pd.DataFrame:
    return pd.read_csv(filename)
