import numpy as np
import pandas as pd
import scipy
from scipy import stats

"""
implement commonly used statistics procedure for genetics,
better support batched operations
"""


def impute_std(geno, mean=None, std=None):
    """
    impute the mean and then standardize
    geno: (num_indivs x num_snps) numpy array
    """
    if mean is None and std is None:
        mean = np.nanmean(geno, axis=0)
        nanidx = np.where(np.isnan(geno))
        geno[nanidx] = mean[nanidx[1]]
        std = np.std(geno, axis=0)
        std_geno = (geno - mean) / std
    else:
        nanidx = np.where(np.isnan(geno))
        geno[nanidx] = mean[nanidx[1]]
        std_geno = (geno - mean) / std
    return std_geno


def mean_std(geno, chunk_size=500):
    row_count = geno.row_count
    col_count = geno.col_count
    mean = np.zeros(col_count)
    std = np.zeros(col_count)

    for i in range(0, col_count, chunk_size):
        sub_geno = geno[:, i : i + chunk_size].read().val
        sub_mean = np.nanmean(sub_geno, axis=0)
        mean[i : i + chunk_size] = sub_mean
        nanidx = np.where(np.isnan(sub_geno))
        sub_geno[nanidx] = sub_mean[nanidx[1]]
        std[i : i + chunk_size] = np.std(sub_geno, axis=0)
    df = pd.DataFrame({"mean": mean, "std": std})
    return df


def cov(geno, mean_std, chunk_size=500):
    """
    geno: a Bed object which we are interested in estimating the LD
    chunk_size: number of SNPs to compute at a time
    """
    # first compute the mean and standard deviation for the given geno
    num_indv, num_snps = geno.shape
    cov = np.zeros([num_snps, num_snps])
    # compute mean and standard deviation
    for row_start in range(0, num_snps, chunk_size):
        for col_start in range(0, num_snps, chunk_size):
            # for each block
            row_stop = row_start + chunk_size
            col_stop = col_start + chunk_size
            if row_stop > num_snps:
                row_stop = num_snps
            if col_stop > num_snps:
                col_stop = num_snps

            std_row_geno = imputed_std(
                geno[:, row_start:row_stop].read().val,
                mean_std["mean"][row_start:row_stop].values,
                mean_std["std"][row_start:row_stop].values,
            )
            std_col_geno = imputed_std(
                geno[:, col_start:col_stop].read().val,
                mean_std["mean"][col_start:col_stop].values,
                mean_std["std"][col_start:col_stop].values,
            )

            cov[
                np.ix_(np.arange(row_start, row_stop), np.arange(col_start, col_stop))
            ] = np.dot(std_row_geno.T, std_col_geno) / (std_row_geno.shape[0])
    return cov


def quad_form(x, A):
    return np.dot(np.dot(x.T, A), x)


def zsc2pval(zsc):
    return 1 - scipy.stats.norm.cdf(zsc)


def pval2zsc(pval):
    return -scipy.stats.norm.ppf(pval)


def chi2_to_logpval(chi2, dof=1):
    return stats.chi2.logsf(chi2, dof)