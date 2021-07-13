__all__ = ["allele_per_anc", "admix_grm", "grm", "get_dependency", "plink2", "gcta"]

from dask.array.core import Array
import numpy as np
import dask.array as da
import xarray as xr
from pandas.api.types import infer_dtype, is_string_dtype, is_categorical_dtype
import dask.array as da
import dask
from ._ext import get_dependency, plink2, gcta


def pca(
    dset: xr.Dataset,
    method: str = "grm",
    n_components: int = 10,
    n_power_iter: int = 4,
    inplace: bool = True,
):
    """
    Calculate PCA of dataset

    Parameters
    ----------
    dset: xr.Dataset
        Dataset to get PCA
    method: str
        Method to calculate PCA, "grm" or "randomized"
    n_components: int
        Number of components to keep
    n_power_iter: int
        Number of power iterations to use for randomized PCA
    inplace: bool
        whether to return a new dataset or modify the input dataset
    """

    assert method in ["grm", "randomized"], "`method` should be 'grm' or 'randomized'"
    if method == "grm":
        if "grm" not in dset.data_vars:
            # calculate grm
            if inplace:
                grm(dset, inplace=True)
                grm_ = dset.data_vars["grm"]
            else:
                grm_ = grm(dset, inplace=False)
        else:
            grm_ = dset.data_vars["grm"]
        # calculate pca
        u, s, v = da.linalg.svd(grm_)
        u, s, v = dask.compute(u, s, v)
        exp_var = (s ** 2) / n_indiv
        full_var = exp_var.sum()
        exp_var_ratio = exp_var / full_var

        coords = u[:, :n_components] * s[:n_components]
        # TODO: unit test with gcta64
    elif method == "randomized":
        # n_indiv, n_snp = gn.shape
        # if copy:
        #     gn = gn.copy()

        # mean_ = gn.mean(axis=0)
        # std_ = gn.std(axis=0)
        # gn -= mean_
        # gn /= std_
        u, s, v = da.linalg.svd_compressed(
            dset, n_components=n_components, n_power_iter=n_power_iter
        )

        # # calculate explained variance
        # exp_var = (s ** 2) / n_indiv
        # full_var = exp_var.sum()
        # exp_var_ratio = exp_var / full_var

        # coords = u[:, :n_components] * s[:n_components]

    # return coords


# def pca(gn, n_components=10, copy=True):
#     # standardize to mean 0 and variance 1
#     # check inputs
#         copy = copy if copy is not None else self.copy
#         gn = asarray_ndim(gn, 2, copy=copy)
#         if not gn.dtype.kind == 'f':
#             gn = gn.astype('f2')

#         # center
#         gn -= self.mean_

#         # scale
#         gn /= self.std_

#     u, s, v = da.linalg.svd_compressed(dset.geno.sum(axis=2).data, k=10, seed=1234)
#     # calculate explained variance
#     self.explained_variance_ = exp_var = (s ** 2) / n_samples
#     full_var = np.var(x, axis=0).sum()
#     self.explained_variance_ratio_ = exp_var / full_var
#             # store components
#     self.components_ = v
#     return u, s, v


def allele_per_anc(ds, return_mask=False, inplace=True):
    """Get allele count per ancestry

    Parameters
    ----------
    ds: xr.Dataset
        Containing geno, lanc, n_anc
    return_mask: bool
        whether to return a masked array
    inplace: bool
        whether to return a new dataset or modify the input dataset
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

    assert isinstance(geno, da.Array) & isinstance(
        lanc, da.Array
    ), "`geno` and `lanc` should be dask array"
    # make sure the chunk size along the haploid axis to be 2
    geno = geno.rechunk({2: 2})
    lanc = lanc.rechunk({2: 2})

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
        rls_allele_per_anc = da.ma.masked_array(
            rls_allele_per_anc, mask=mask, fill_value=0
        )
    if inplace:
        ds["allele_per_anc"] = xr.DataArray(
            rls_allele_per_anc, dims=("indiv", "snp", "anc")
        )
    else:
        return rls_allele_per_anc


def grm(dset: xr.Dataset, method="gcta", inplace=True):
    """Calculate the GRM matrix
    The GRM matrix is calculated treating the genotypes as from one ancestry population,
    the same as GCTA.

    Parameters
    ----------
    dset: xr.Dataset
        dataset containing geno
    method: str
        method to calculate the GRM matrix, `gcta` or `raw`
        - `raw`: use the raw genotype data without any transformation
        - `center`: center the genotype data only
        - `gcta`: use the GCTA implementation of GRM, center + standardize
    inplace: bool
        whether to return a new dataset or modify the input dataset
    Returns
    -------
    n_indiv x n_indiv GRM matrix if `inplace` is False, else return None
    """

    assert method in [
        "raw",
        "center",
        "gcta",
    ], "`method` should be `raw`, `center`, or `gcta`"
    g = dset["geno"].data
    n_indiv, n_snp, n_haplo = g.shape
    g = g.sum(axis=2)

    if method == "raw":
        grm = np.dot(g, g.T) / n_snp
    elif method == "center":
        g -= g.mean(axis=0)
        grm = np.dot(g, g.T) / n_snp
    elif method == "gcta":
        # normalization
        g_mean = g.mean(axis=0)
        assert np.all((0 < g_mean) & (g_mean < 2)), "for some SNP, MAF = 0"
        g = (g - g_mean) / np.sqrt(g_mean * (2 - g_mean) / 2)
        # calculate GRM
        grm = np.dot(g, g.T) / n_snp
    else:
        raise ValueError("method should be `gcta` or `raw`")

    if inplace:
        dset["grm"] = xr.DataArray(grm, dims=("indiv", "indiv"))
    else:
        return grm


def admix_grm(dset, center: bool = False, mask: bool = False, inplace=True):
    """Calculate ancestry specific GRM matrix

    Parameters
    ----------
    center: bool
        whether to center the `allele_per_ancestry` matrix
        in the calculation
    mask: bool
        whether to mask the missing values when perform the
        centering
    inplace: bool
        whether to return a new dataset or modify the input dataset

    Returns
    -------
    If `inplace` is False, return a dictionary of GRM matrices
        - K1: np.ndarray
            ancestry specific GRM matrix for the 1st ancestry
        - K2: np.ndarray
            ancestry specific GRM matrix for the 2nd ancestry
        - K12: np.ndarray
            ancestry specific GRM matrix for cross term of the 1st and 2nd ancestry

    If `inplace` is True, return None
        "admix_grm_K1", "admix_grm_K2", "admix_grm_K12" will be added to the dataset
    """

    geno = dset["geno"].data
    lanc = dset["lanc"].data
    n_anc = dset.attrs["n_anc"]
    assert n_anc == 2, "only two-way admixture is implemented"
    assert np.all(geno.shape == lanc.shape)

    apa = allele_per_anc(dset, return_mask=mask, inplace=False).astype(float)

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

    if inplace:
        dset["admix_grm_K1"] = xr.DataArray(K1, dims=("indiv", "indiv"))
        dset["admix_grm_K2"] = xr.DataArray(K2, dims=("indiv", "indiv"))
        dset["admix_grm_K12"] = xr.DataArray(K12, dims=("indiv", "indiv"))
        return None
    else:
        return {"K1": K1, "K2": K2, "K12": K12}