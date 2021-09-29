"""Common utility functions for `admix`. Used by more than 2 modules"""
import admix
import os
import subprocess
from contextlib import contextmanager
import os
import dask.array as da
import numpy as np
from tqdm import tqdm
from scipy import stats


@contextmanager
def cd(newdir):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)


def get_cache_dir() -> str:
    """Get the cache directory for admix-kit

    Returns
    -------
    [type]
        [description]
    """
    cache_dir = os.path.join(os.path.dirname(admix.__file__), "../.admix_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _impute_with_mean(geno, inplace=False):
    """impute the each entry using the mean of each column

    Parameters
    ----------
    geno : np.ndarray
        (n_indiv, n_snp) genotype matrix

    Returns
    -------
    if inplace:
        geno : np.ndarray
            (n_indiv, n_snp) genotype matrix
    else:
        None
    """
    if not inplace:
        geno = geno.copy()

    mean = np.nanmean(geno, axis=0)
    nanidx = np.where(np.isnan(geno))
    geno[nanidx] = mean[nanidx[1]]

    if not inplace:
        return geno
    else:
        return None


def _geno_mult_mat(
    geno: da.Array,
    mat: np.ndarray,
    impute_geno: bool = True,
    transpose_geno: bool = False,
    return_snp_var: bool = False,
) -> np.ndarray:
    """Multiply genotype matrix with a matrix

    Chunk of genotype matrix will be read sequentially along the SNP dimension,
    and multiplied with the `mat`.

    Without transpose, result will be (n_indiv, n_rep)
    With transpose, result will be (n_snp, n_rep)

    Missing values in geno will be imputed with the mean of the genotype matrix.

    Parameters
    ----------
    geno : da.Array
        Genotype matrix with shape (n_indiv, n_snp)
        geno.chunk contains the chunk of genotype matrix to be multiplied
    mat : np.ndarray
        Matrix to be multiplied with the genotype matrix
    impute_geno : bool
        Whether to impute missing values with the mean of the genotype matrix
    transpose_geno : bool
        Whether to transpose the genotype matrix and calulate geno.T @ mat
    return_snp_var : bool
        Whether to return the variance of each SNP, useful in simple linear
        regression

    Returns
    -------
    np.ndarray
        Result of the multiplication
    """
    chunks = geno.chunks[1]
    indices = np.insert(np.cumsum(chunks), 0, 0)
    n_indiv, n_snp = geno.shape
    n_rep = mat.shape[1]

    snp_var = np.zeros(n_snp)
    if not transpose_geno:
        assert (
            mat.shape[0] == n_snp
        ), "when transpose_geno is False, matrix should be of shape (n_snp, n_rep)"
        ret = np.zeros((n_indiv, n_rep))
        for i in tqdm(range(len(indices) - 1), desc="_geno_mult_mat"):
            start, stop = indices[i], indices[i + 1]
            geno_chunk = geno[:, start:stop].compute()
            # impute missing genotype
            if impute_geno:
                _impute_with_mean(geno_chunk, inplace=True)
            ret += np.dot(geno_chunk, mat[start:stop, :])

            if return_snp_var:
                snp_var[start:stop] = np.var(geno_chunk, axis=0)
    else:
        # genotype is transposed
        assert (
            mat.shape[0] == n_indiv
        ), "when transpose_geno is True, matrix should be of shape (n_indiv, n_rep)"
        ret = np.zeros((n_snp, n_rep))
        for i in tqdm(range(len(indices) - 1), desc="_geno_mult_mat"):
            start, stop = indices[i], indices[i + 1]
            geno_chunk = geno[:, start:stop].compute()
            # impute missing genotype
            if impute_geno:
                _impute_with_mean(geno_chunk, inplace=True)
            ret[start:stop, :] = np.dot(geno_chunk.T, mat)

            if return_snp_var:
                snp_var[start:stop] = np.var(geno_chunk, axis=0)

    if return_snp_var:
        return ret, snp_var
    else:
        return ret


def pval2chisq(pval: np.ndarray, two_sided: bool = True):
    """Convert p-value to z-score

    Parameters
    ----------
    pval : np.ndarray
        p-value
    two_sided : bool
        Whether to use two-sided test

    Returns
    -------
    np.ndarray
        z-score
    """
    if two_sided:
        return stats.norm.ppf(pval / 2) ** 2
    else:
        return stats.norm.ppf(pval) ** 2


def zsc2pval(zscore: np.ndarray, two_sided: bool = True):
    """Convert z-score to p-value

    Parameters
    ----------
    zscore : np.ndarray
        z-score
    two_sided : bool
        Whether to use two-sided test

    Returns
    -------
    np.ndarray
        p-value
    """
    if two_sided:
        return 2 * (1 - stats.norm.cdf(np.abs(zscore)))
    else:
        return 1 - stats.norm.cdf(zscore)