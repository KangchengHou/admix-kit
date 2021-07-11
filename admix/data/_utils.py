import numpy as np
import re
import dask.array as da
import pandas as pd
import xarray as xr
import warnings
from pandas.api.types import infer_dtype, is_string_dtype, is_categorical_dtype
from os.path import dirname, join
from typing import List


def make_dataset(
    geno, snp: pd.DataFrame, indiv: pd.DataFrame, meta: dict = None, lanc=None
):
    """Make up a dataset

    Parameters
    ----------
    geno : (#indiv, #snp, 2) array_like
        Genotype count matrix
    snp : pd.DataFrame
        Index is identifier to SNP
        each column corresponds to an attribute
    indiv : pd.DataFrame
        Index is identifier to individuals
        each column corresponds to an attribute
    meta : dict, optional
        Meta information about the dataset
    lanc : (#indiv, #snp, 2) array_like, optional
        Local ancestry
    """
    # fill `data_vars`
    data_vars = {
        "geno": (("indiv", "snp", "haploid"), geno),
    }
    if lanc is not None:
        data_vars["lanc"] = (("indiv", "snp", "haploid"), lanc)

    coords = {}
    # fill SNP information
    coords["snp"] = snp.index.values
    if not is_string_dtype(coords["snp"]):
        warnings.warn("Transforming snp index to str")
    coords["snp"] = coords["snp"].astype(str)

    for col in snp.columns:
        vals = snp[col].values
        if is_string_dtype(snp[col]):
            vals = snp[col].values.astype(str)

        coords[f"{col}@snp"] = ("snp", vals)

    # fill in individual information
    coords["indiv"] = indiv.index.values
    if not is_string_dtype(coords["indiv"]):
        warnings.warn("Transforming indiv index to str")
    coords["indiv"] = coords["indiv"].astype(str)

    for col in indiv.columns:
        vals = indiv[col].values
        if is_string_dtype(indiv[col]):
            vals = vals.astype(str)
        coords[f"{col}@indiv"] = ("indiv", vals)

    dset = xr.Dataset(data_vars=data_vars, coords=coords, attrs=meta)
    return dset


def seperate_ld_blocks(anc, phgeno, legend, ld_blocks):
    assert len(legend) == anc.shape[1]
    assert len(legend) == phgeno.shape[1]

    rls_list = []
    for block_i, block in ld_blocks.iterrows():
        block_index = np.where(
            (block.START <= legend.position) & (legend.position < block.STOP)
        )[0]
        block_legend = legend.loc[block_index]
        block_anc = anc[:, block_index]
        block_phgeno = phgeno[:, block_index]
        rls_list.append((block_anc, block_phgeno, block_legend))
    return rls_list


def convert_anc_count(phgeno: np.ndarray, anc: np.ndarray) -> np.ndarray:
    """
    Convert from ancestry and phased genotype to number of minor alles for each ancestry
    version 2, it should lead to exact the same results as `convert_anc_count`

    Args:
        phgeno (np.ndarray): (n_indiv, 2 x n_snp), the first half columns contain the first haplotype,
            the second half columns contain the second haplotype
        anc (np.ndarray): n_indiv x 2n_snp, match `phgeno`

    Returns:
        np.ndarray: n_indiv x 2n_snp, the first half columns stores the number of minor alleles
        from the first ancestry, the second half columns stores the number of minor
        alleles from the second ancestry
    """
    n_indiv = anc.shape[0]
    n_snp = anc.shape[1] // 2
    n_anc = 2
    geno = np.zeros_like(phgeno)
    for haplo_i in range(2):
        haplo_slice = slice(haplo_i * n_snp, (haplo_i + 1) * n_snp)
        haplo_phgeno = phgeno[:, haplo_slice]
        haplo_anc = anc[:, haplo_slice]
        for anc_i in range(n_anc):
            geno[:, (anc_i * n_snp) : ((anc_i + 1) * n_snp)][
                haplo_anc == anc_i
            ] += haplo_phgeno[haplo_anc == anc_i]

    return geno


def load_toy() -> List[xr.Dataset]:
    """Load toy dataset

    Load simulated
    (1) 50 admixed individuals
    (2) 50 EUR individuals
    (3) 50 AFR individuals

    5000 SNPs

    Returns
    -------
    List[xr.Dataset]
        [dset_admix, dset_eur, dset_afr]
    """

    module_path = dirname(__file__)
    test_data_path = join(module_path, "../../tests/test-data")
    dset_eur = xr.open_zarr(join(test_data_path, "eur.zip"))
    dset_afr = xr.open_zarr(join(test_data_path, "afr.zip"))
    dset_admix = xr.open_zarr(join(test_data_path, "admix.zip"))
    return dset_admix, dset_eur, dset_afr


def load_lab_dataset(name: str) -> xr.Dataset:
    """Load prepared dataset in Bogdan lab, currently available
    if you use this function on cluster, and have access to
    `/u/project/pasaniuc/pasaniucdata/admixture/dataset`

    - `simulate_eur_afr.20_80`: Simulated admixture of 20% EUR and 80% AFR
    - `simulate_eur_afr.50_50`: Simulated admixture of 50% EUR and 50% AFR
    - `ukb_eur_afr`: Admixed individuals in UK Biobank

    Returns
    -------
    xr.Dataset
        dataset
    """
    assert name in ["simulate_eur_afr.20_80", "simulate_eur_afr.50_50", "ukb_eur_afr"]
    root_dir = "/u/project/pasaniuc/pasaniucdata/admixture/dataset"
    assert os.path.isdir(root_dir), f"check that you have access to {root_dir}"

    dset = xr.open_zarr(join(root_dir, name + ".zip"))
    return dset


def convert_anc_count2(phgeno, anc):
    """
    Convert from ancestry and phased genotype to number of minor alles for each ancestry

    Args
    ----
    phgeno: n_indiv x 2n_snp, the first half columns contain the first haplotype,
        the second half columns contain the second haplotype
    anc: n_indiv x 2n_snp, match `phgeno`

    Returns
    ----
    geno: n_indiv x 2n_snp, the first half columns stores the number of minor alleles
        from the first ancestry, the second half columns stores the number of minor
        alleles from the second ancestry
    """
    n_indiv = anc.shape[0]
    n_snp = anc.shape[1] // 2
    phgeno = phgeno.reshape((n_indiv * 2, n_snp))
    anc = anc.reshape((n_indiv * 2, n_snp))

    geno = np.zeros((n_indiv, n_snp * 2), dtype=np.int8)
    for indiv_i in range(n_indiv):
        for haplo_i in range(2 * indiv_i, 2 * indiv_i + 2):
            for anc_i in range(2):
                anc_snp_index = np.where(anc[haplo_i, :] == anc_i)[0]
                geno[indiv_i, anc_snp_index + anc_i * n_snp] += phgeno[
                    haplo_i, anc_snp_index
                ]
    return geno


def add_up_haplotype(haplo):
    """
    Adding up the values from two haplotypes

    Args
    -----
    haplo: (n_indiv, 2 * n_snp) matrix

    Returns
    -----
    (n_indiv, n_snp) matrix with added up haplotypes
    """
    assert haplo.shape[1] % 2 == 0
    n_snp = haplo.shape[1] // 2
    return haplo[:, np.arange(n_snp)] + haplo[:, np.arange(n_snp) + n_snp]
