import numpy as np
from numpy import (
    float32,
    tril_indices_from,
)
import pandas as pd


def write_gcta_grm(
    file_prefix: str, grm: np.ndarray, df_id: pd.DataFrame, n_snps: np.ndarray
) -> None:
    """
    Write a GCTA grm file.

    Parameters
    ----------
    file_prefix : str
        The prefix of the file to write.
    grm : array_like, shape (n_snps, n_snps)
        The GRM matrix.
    df_id : array_like, shape (n_indiv,)
        The ID of each individual.
    n_snps : np.ndarray, shape (n_indiv,)
        The number of SNPs for each individual.

    Returns
    -------
    None
    """

    bin_file = file_prefix + ".grm.bin"
    N_file = file_prefix + ".grm.N.bin"
    id_file = file_prefix + ".grm.id"

    # id
    df_id.to_csv(id_file, sep="\t", header=None, index=False)
    # bin
    grm[tril_indices_from(grm)].astype(float32).tofile(bin_file)
    # N
    n_snps.astype(float32).tofile(N_file)