import numpy as np
import re
import dask.array as da
import pandas as pd
import xarray as xr
import warnings
from pandas.api.types import infer_dtype, is_string_dtype, is_categorical_dtype


def allele_per_anc(ds, return_mask=False):
    """Get allele count per ancestry

    Parameters
    ----------
    ds: xr.Dataset
        Containing geno, lanc, n_anc
    return_mask: bool
        whether to return a masked array

    Returns
    -------
    Return allele counts per ancestries
    """
    geno, lanc = ds.data_vars["geno"].data, ds.data_vars["lanc"].data
    n_anc = ds.attrs["n_anc"]
    assert np.all(geno.shape == lanc.shape), "shape of `hap` and `lanc` are not equal"
    assert geno.ndim == 3, "`hap` and `lanc` should have three dimension"
    n_indiv, n_snp, n_haplo = geno.shape
    assert n_haplo == 2, "`n_haplo` should equal to 2, check your data"

    if isinstance(geno, da.Array):
        assert isinstance(lanc, da.Array)
        # make sure the chunk size along the haploid axis to be 2
        geno = geno.rechunk({2: 2})
        lanc = lanc.rechunk({2: 2})
    else:
        assert isinstance(geno, np.ndarray) & isinstance(lanc, np.ndarray)

    def helper(geno_chunk, lanc_chunk, n_anc):
        n_indiv, n_snp, n_haplo = geno_chunk.shape
        apa = np.zeros((n_indiv, n_snp, n_anc), dtype=np.int8)

        for i_haplo in range(n_haplo):
            haplo_hap = geno_chunk[:, :, i_haplo]
            haplo_lanc = lanc_chunk[:, :, i_haplo]
            for i_anc in range(n_anc):
                apa[:, :, i_anc][haplo_lanc == i_anc] += haplo_hap[haplo_lanc == i_anc]
        return apa

    rls_allele_per_anc = da.map_blocks(
        lambda a, b: helper(a, b, n_anc=n_anc), geno, lanc
    )
    if return_mask:
        mask = np.dstack([np.all(lanc != i_anc, axis=2) for i_anc in range(n_anc)])
        return da.ma.masked_array(rls_allele_per_anc, mask=mask, fill_value=0)
    else:
        return rls_allele_per_anc


def admix_grm(dset, center: bool = False, mask: bool = False) -> dict:
    """Calculate ancestry specific GRM matrix

    Parameters
    ----------
    center: bool
        whether to center the `allele_per_ancestry` matrix
        in the calculation
    mask: bool
        whether to mask the missing values when perform the
        centering
    Returns
    -------
    A dictionary containing the GRM matrices
        - K1: np.ndarray
            ancestry specific GRM matrix for the 1st ancestry
        - K2: np.ndarray
            ancestry specific GRM matrix for the 2nd ancestry
        - K12: np.ndarray
            ancestry specific GRM matrix for cross term of the 1st and 2nd ancestry
    """

    geno = dset["geno"].data
    lanc = dset["lanc"].data
    n_anc = dset.attrs["n_anc"]
    assert n_anc == 2, "only two-way admixture is implemented"
    assert np.all(geno.shape == lanc.shape)

    apa = allele_per_anc(dset, return_mask=mask).astype(float)

    n_indiv, n_snp = apa.shape[0:2]

    if center:
        if mask:
            # only calculate at nonmissing entries
            mean_apa = np.ma.getdata(np.mean(apa, axis=0).compute())
            apa = apa - mean_apa
            apa = da.ma.getdata(apa)
        else:
            # calculate at all entries
            mean_apa = np.mean(da.ma.getdata(apa), axis=0).compute()
            apa = da.ma.getdata(apa) - mean_apa

    a1, a2 = apa[:, :, 0], apa[:, :, 1]

    K1 = np.dot(a1, a1.T) / n_snp
    K2 = np.dot(a2, a2.T) / n_snp
    K12 = np.dot(a1, a2.T) / n_snp
    return {
        "K1": K1,
        "K2": K2,
        "K12": K12,
    }