import pandas as pd
import random
from scipy import stats

def stratify_calculate_r2(
    df: pd.DataFrame, x_col: str, y_col: str, stratify_col: str, n_level: int = 5, n_sample: int = 10,  sample_prop: float = 0.8
) -> pd.DataFrame:
    """
    Stratify dataframe by `stratify_col` with `n_level` levels; within each level of
    the stratification, calculate the R2 value of the regression of `y_col` on `x_col`.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the data, must have columns `x_col`, `y_col` and
        `stratify_col`.
    x_col : str
        Name of the column containing the predictor variable.
    y_col : str
        Name of the column containing the response variable.
    stratify_col : str
        Name of the column containing the stratification variable.
    n_level : int
        Number of levels to stratify.

    n_sample : int
        Number of samplying to do.
    
    sample_prop : float
        Proportion of sample size to population size

    Returns
    -------
    pandas.DataFrame
        Dataframe with `n_level` rows and two columns `R2` and `R2_std`
        - R2: R2 between x_col and y_col within each level
        - R2_std: standard deviation of R2 within each level obtained with bootstrap

    References
    ----------
    For bootstrap standard deviation, see:
    https://stats.stackexchange.com/questions/172662/how-do-you-calculate-the-standard-error-of-r2
    """

    df = df.dropna(subset=['x_col', 'y_col'])

    level_li = [i*(1/n_level) for i in range(n_level+1)]
    qtl_li = df['stratify_col'].quantile(level_li).values
    grp_index_li = []
    for grp_i in range(n_level-1):
        grp_index_li.append(df.index[(qtl_li[grp_i] <= df['stratify_col']) & (df['stratify_col'] < qtl_li[grp_i+1])])

    grp_index_li.append(df.index[(qtl_li[n_level-1] <= df['stratify_col']) & (df['stratify_col'] <= qtl_li[n_level])])

    grp_r2_samples = []
    for i_sample in range(n_sample):
        grp_index_random = []
        grp_r2_li = []
        for i in range(n_level):
            grp_index_random.append(random.choices(grp_index_li[i], k=int(0.8*grp_index_li[i].shape[0])))
            res = stats.linregress(df.loc[grp_index_random[i], 'x_col'], df.loc[grp_index_random[i], 'y_col'])
            grp_r2_li.append(res.rvalue**2)
        grp_r2_samples.append(grp_r2_li)
    
    avg_grp_r2 = np.mean(grp_r2_samples, axis=0)
    std_grp_r2 = np.std(grp_r2_samples, axis=0)
    d = {'level': range(1, n_level+1), 'R2': avg_grp_r2, 'R2_std': std_grp_r2}
    output = pd.DataFrame(data=d)
    return output
