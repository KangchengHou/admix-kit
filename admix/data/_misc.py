import pandas as pd
from typing import List
import admix


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
        admix.logger.info(f"Detected categorical columns: {','.join(cols)}")

    added_cols = []
    df = df.copy()
    for col in df:
        # create study dummies variables
        dummies = pd.get_dummies(df[col], drop_first=True)
        dummies.columns = [f"{col}_{s}" for s in dummies.columns]
        added_cols.extend(dummies.columns)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(columns=[col])
    admix.logger.info(f"Added dummy variables: {','.join(added_cols)}")
    return df