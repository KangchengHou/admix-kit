from numpy import (
    asarray,
    float32,
    float64,
    fromfile,
    int64,
    tril,
    tril_indices_from,
    zeros,
)
from pandas import read_csv


def read_gcta_grm(file_prefix) -> dict:
    """
    Reads the GRM from a GCTA formated file.

    Parameters
    ----------
    file_prefix : str
        The prefix of the GRM to be read.

    Returns
    -------
    dict
        A dictionary with the GRM values.
        - grm: GRM matrix
        - df_id: ids of the individuals
        - n_snps: number of SNP

    """

    bin_file = file_prefix + ".grm.bin"
    N_file = file_prefix + ".grm.N.bin"
    id_file = file_prefix + ".grm.id"

    df_id = read_csv(id_file, sep="\t", header=None, names=["sample_0", "sample_1"])
    n = df_id.shape[0]
    k = asarray(fromfile(bin_file, dtype=float32), float64)
    n_snps = asarray(fromfile(N_file, dtype=float32), int64)

    K = zeros((n, n))
    K[tril_indices_from(K)] = k
    K = K + tril(K, -1).T
    return {
        "grm": K,
        "df_id": df_id,
        "n_snps": n_snps,
    }