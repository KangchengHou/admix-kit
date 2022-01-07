import pandas as pd
import random
from scipy import stats


def stratify_calculate_r2(
    df: pd.DataFrame, x_col: str, y_col: str, stratify_col: str, n_level: int = 5
) -> pd.DataFrame:
    pass