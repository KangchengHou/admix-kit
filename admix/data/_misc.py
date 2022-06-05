import pandas as pd
from typing import List
import admix
import numpy as np


def convert_dummy(df: pd.DataFrame, cols: List[str] = None) -> pd.DataFrame:
    """
    Convert dummy variables to categorical variables for each column in df.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to convert
    cols : List[str], optional
        Columns to convert, by default None (columns are selected automatically)

    Returns
    -------
    pd.DataFrame
        Converted dataframe
    """
    if cols is None:
        cols = list(set(df.columns) - set(df._get_numeric_data().columns))

    added_cols = []
    df = df.copy()
    for col in cols:
        # create study dummies variables
        dummies = pd.get_dummies(df[col], drop_first=True)
        dummies.columns = [f"{col}_{s}" for s in dummies.columns]
        dummies.loc[df[col].isnull(), dummies.columns] = np.nan
        added_cols.extend(dummies.columns)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(columns=[col])
    if len(added_cols) > 0:
        admix.logger.info(
            f"Detected categorical columns: {','.join(cols)}, "
            f"and dded dummy variables: {','.join(added_cols)}"
        )
    return df