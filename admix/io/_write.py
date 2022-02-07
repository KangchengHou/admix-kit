import numpy as np
from numpy import (
    float32,
    tril_indices_from,
)
import pandas as pd
import dask.array as da
import admix
from typing import Union
import dapgen


def write_dataset(
    geno: da.Array,
    lanc: admix.data.Lanc,
    df_indiv: pd.DataFrame,
    df_snp: pd.DataFrame,
    out_prefix: str,
) -> None:

    n_snp, n_indiv = geno.shape[0:2]
    assert lanc.n_snp == n_snp
    assert lanc.n_indiv == n_indiv
    assert df_indiv.shape[0] == n_indiv
    assert df_snp.shape[0] == n_snp

    # write genotype
    dapgen.write_pgen(f"{out_prefix}.pgen", geno)

    # write local ancestry
    admix.io.write_lanc(f"{out_prefix}.lanc", lanc)

    # write df_indiv
    df_indiv.rename_axis("#IID").to_csv(f"{out_prefix}.psam", sep="\t")

    # write df_snp
    df_snp = df_snp.reset_index().rename(columns={"snp": "ID", "CHROM": "#CHROM"})

    FIXED_COLS = ["#CHROM", "POS", "ID", "REF", "ALT"]
    df_snp = df_snp[
        FIXED_COLS + [col for col in df_snp.columns if col not in FIXED_COLS]
    ]

    df_snp.to_csv(f"{out_prefix}.pvar", sep="\t", index=False)


def write_lanc(path: str, lanc: Union[np.ndarray, da.Array, admix.data.Lanc]) -> None:
    """
    Write local ancestry matrix to file.

    Parameters
    ----------
    path : str
        The path to the file to write.
    lanc : array_like, shape (n_snps, n_snps)
        The local ancestry matrix.
    """
    if isinstance(lanc, np.ndarray) or isinstance(lanc, da.Array):
        lanc = admix.data.Lanc(array=lanc)

    assert isinstance(
        lanc, admix.data.Lanc
    ), "lanc must be a numpy array or a dask array, or admix.data.Lanc"
    lanc.write(path)


def write_digit_mat(path, mat):
    """
    Read a matrix of integer with [0-9], and with no delimiter.

    Args
    ----

    """
    np.savetxt(path, mat, fmt="%d", delimiter="")


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


# def write_gcta_pheno(
#     file_prefix: str, df_pheno: pd.DataFrame, df_id: pd.DataFrame
# ) -> None:
#     """
#     Write a GCTA phenotype file.

#     Parameters
#     ----------
#     file_prefix : str
#         The prefix of the file to write.
#     df_pheno : array_like, shape (n_indiv, n_pheno)
#         The phenotype matrix.
#     df_id : array_like, shape (n_indiv,)
#         The ID of each individual.

#     Returns
#     -------
#     None
#     """

#     pheno_file = file_prefix + ".pheno"
#     id_file = file_prefix + ".id"

#     # id
#     df_id.to_csv(id_file, sep="\t", header=None, index=False)
#     # pheno
#     df_pheno.to_csv(pheno_file, sep="\t", header=None, index=False)