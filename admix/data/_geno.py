import numpy as np
from tqdm import tqdm
import dask.array as da
import xarray as xr


def calc_snp_prior_var(df_snp_info, her_model):
    """
    Calculate the SNP prior variance from SNP information
    """
    assert her_model in ["uniform", "gcta", "ldak", "mafukb"]
    if her_model == "uniform":
        return np.ones(len(df_snp_info))
    elif her_model == "gcta":
        freq = df_snp_info["FREQ"].values
        assert np.all(freq > 0), "frequencies should be larger than zero"
        return np.float_power(freq * (1 - freq), -1)
    elif her_model == "mafukb":
        # MAF-dependent genetic architecture, \alpha = -0.38 estimated from meta-analysis in UKB traits
        freq = df_snp_info["FREQ"].values
        assert np.all(freq > 0), "frequencies should be larger than zero"
        return np.float_power(freq * (1 - freq), -0.38)
    elif her_model == "ldak":
        freq, weight = df_snp_info["FREQ"].values, df_snp_info["LDAK_WEIGHT"].values
        return np.float_power(freq * (1 - freq), -0.25) * weight
    else:
        raise NotImplementedError


def impute_with_mean(geno, inplace=False):
    """impute the each entry using the mean of each row

    Parameters
    ----------
    geno : np.ndarray
        (n_snp, n_indiv) genotype matrix

    Returns
    -------
    if inplace:
        geno : np.ndarray
            (n_snp, n_indiv) genotype matrix
    else:
        None
    """
    if not inplace:
        geno = geno.copy()

    # impute the missing genotypes with the mean of each row
    mean = np.nanmean(geno, axis=1)
    nanidx = np.where(np.isnan(geno))
    # index the mean using the nanidx[0] (row-wise)
    geno[nanidx] = mean[nanidx[0]]

    if not inplace:
        return geno
    else:
        return None


def geno_mult_mat(
    geno: da.Array,
    mat: np.ndarray,
    impute_geno: bool = True,
    mat_dim: str = "snp",
    return_snp_var: bool = False,
) -> np.ndarray:
    """Multiply genotype matrix with a matrix

    Chunk of genotype matrix will be read sequentially along the SNP dimension,
    and multiplied with the `mat`.

    Without transpose, result will be (n_snp, n_rep)
    With transpose, result will be (n_indiv, n_rep)

    Missing values in geno will be imputed with the mean of the genotype matrix.

    Parameters
    ----------
    geno : da.Array
        Genotype matrix with shape (n_snp, n_indiv)
        geno.chunk contains the chunk of genotype matrix to be multiplied
    mat : np.ndarray
        Matrix to be multiplied with the genotype matrix
    impute_geno : bool
        Whether to impute missing values with the mean of the genotype matrix
    mat_dim : str
        First dimension of the `mat`, either "snp" or "indiv"
        Whether to transpose the genotype matrix and calulate geno.T @ mat
    return_snp_var : bool
        Whether to return the variance of each SNP, useful in simple linear
        regression

    Returns
    -------
    np.ndarray
        Result of the multiplication
    """
    assert mat_dim in ["snp", "indiv"], "mat_dim should be `snp` or `indiv`"

    # chunks over SNPs
    chunks = geno.chunks[0]
    indices = np.insert(np.cumsum(chunks), 0, 0)
    n_snp, n_indiv = geno.shape
    n_rep = mat.shape[1]

    snp_var = np.zeros(n_snp)
    if mat_dim == "indiv":
        # geno: (n_snp, n_indiv)
        # mat: (n_indiv, n_rep)
        assert (
            mat.shape[0] == n_indiv
        ), "when mat_dim is 'indiv', matrix should be of shape (n_indiv, n_rep)"
        ret = np.zeros((n_snp, n_rep))
        for i in tqdm(range(len(indices) - 1), desc="admix.data.geno_mult_mat"):
            start, stop = indices[i], indices[i + 1]
            geno_chunk = geno[start:stop, :].compute()
            # impute missing genotype
            if impute_geno:
                impute_with_mean(geno_chunk, inplace=True)
            ret[start:stop, :] = np.dot(geno_chunk, mat)

            if return_snp_var:
                snp_var[start:stop] = np.var(geno_chunk, axis=0)
    elif mat_dim == "snp":
        # geno: (n_indiv, n_snp)
        # mat: (n_snp, n_rep)
        assert (
            mat.shape[0] == n_snp
        ), "when mat_dim is 'snp', matrix should be of shape (n_snp, n_rep)"
        ret = np.zeros((n_indiv, n_rep))
        for i in tqdm(range(len(indices) - 1), desc="admix.data.geno_mult_mat"):
            start, stop = indices[i], indices[i + 1]
            geno_chunk = geno[start:stop, :].compute()
            # impute missing genotype
            if impute_geno:
                impute_with_mean(geno_chunk, inplace=True)
            ret += np.dot(geno_chunk.T, mat[start:stop, :])

            if return_snp_var:
                snp_var[start:stop] = np.var(geno_chunk, axis=0)
    else:
        raise ValueError("mat_dim should be `snp` or `indiv`")
    if return_snp_var:
        return ret, snp_var
    else:
        return ret


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


def admix_grm(dset, center: bool = False, inplace=True):
    """Calculate ancestry specific GRM matrix

    Parameters
    ----------
    center: bool
        whether to center the `allele_per_ancestry` matrix
        in the calculation
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
    # TODO: everytime should we recompute? or use the cached version?
    # how to make sure the cache is up to date?
    apa = allele_per_anc(dset, center=center).astype(float)

    n_indiv, n_snp = apa.shape[0:2]

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


def af_per_anc(geno, lanc, n_anc=2) -> np.ndarray:
    """
    Calculate allele frequency per ancestry
    Parameters
    ----------
    dset: xr.Dataset
        Containing geno, lanc, n_anc
    Returns
    -------
    List[np.ndarray]
        `n_anc` length list of allele frequencies.
    """
    assert np.all(geno.shape == lanc.shape)
    n_snp = geno.shape[0]
    af = np.zeros((n_snp, n_anc))

    snp_chunks = geno.chunks[0]
    indices = np.insert(np.cumsum(snp_chunks), 0, 0)

    # TODO: replace with the admix_genet_cor implementation
    for i in range(len(indices) - 1):
        start, stop = indices[i], indices[i + 1]
        geno_chunk = geno[start:stop, :, :].compute()
        lanc_chunk = lanc[start:stop, :, :].compute()

        for anc_i in range(n_anc):
            # mask SNPs with local ancestry not `i_anc`
            af[start:stop, anc_i] = (
                np.ma.masked_where(lanc_chunk != anc_i, geno_chunk)
                .mean(axis=(1, 2))
                .data
            )
    return af


def allele_per_anc(geno, lanc, center=False, n_anc=2):
    """Get allele count per ancestry

    Parameters
    ----------
    ds: xr.Dataset
        Containing geno, lanc, n_anc
    center: bool
        whether to center the data around empirical frequencies of each ancestry
    inplace: bool
        whether to return a new dataset or modify the input dataset
    Returns
    -------
    Return allele counts per ancestries
    """
    assert center is False, "center=True should not be used"
    assert np.all(geno.shape == lanc.shape), "shape of `hap` and `lanc` are not equal"
    assert geno.ndim == 3, "`hap` and `lanc` should have three dimension"
    n_snp, n_indiv, n_haplo = geno.shape
    assert n_haplo == 2, "`n_haplo` should equal to 2, check your data"

    assert isinstance(geno, da.Array) & isinstance(
        lanc, da.Array
    ), "`geno` and `lanc` should be dask array"

    # make sure the chunk size along the ploidy axis to be 2
    geno = geno.rechunk({2: 2})
    lanc = lanc.rechunk({2: 2})

    # rechunk so that all chunk of `n_anc` is passed into the helper function
    assert (
        n_anc == 2
    ), "`n_anc` should be 2, NOTE: not so clear what happens when `n_anc = 3`"

    assert (
        geno.chunks == lanc.chunks
    ), "`geno` and `lanc` should have the same chunk size"

    assert (
        len(geno.chunks[1]) == 1
    ), "geno / lanc should not be chunked across individual dimension"

    def helper(geno_chunk, lanc_chunk, n_anc, center):

        n_snp, n_indiv, n_haplo = geno_chunk.shape
        if center:
            af_chunk = af_per_anc(
                da.from_array(geno_chunk), da.from_array(lanc_chunk), n_anc=n_anc
            )
        else:
            af_chunk = None
        apa = np.zeros((n_snp, n_indiv, n_anc), dtype=np.float64)
        for i_haplo in range(n_haplo):
            haplo_hap = geno_chunk[:, :, i_haplo]
            haplo_lanc = lanc_chunk[:, :, i_haplo]
            for i_anc in range(n_anc):
                if af_chunk is None:
                    apa[:, :, i_anc][haplo_lanc == i_anc] += haplo_hap[
                        haplo_lanc == i_anc
                    ]
                else:
                    # for each SNP, find the corresponding allele frequency
                    apa[:, :, i_anc][haplo_lanc == i_anc] += (
                        haplo_hap[haplo_lanc == i_anc]
                        - af_chunk[np.where(haplo_lanc == i_anc)[0], i_anc]
                    )
        return apa

    rls_allele_per_anc = da.map_blocks(
        lambda geno_chunk, lanc_chunk: helper(
            geno_chunk=geno_chunk, lanc_chunk=lanc_chunk, n_anc=n_anc, center=center
        ),
        geno,
        lanc,
        dtype=np.float64,
    )

    return rls_allele_per_anc


# def pca(
#     dset: xr.Dataset,
#     method: str = "grm",
#     n_components: int = 10,
#     n_power_iter: int = 4,
#     inplace: bool = True,
# ):
#     """
#     Calculate PCA of dataset

#     Parameters
#     ----------
#     dset: xr.Dataset
#         Dataset to get PCA
#     method: str
#         Method to calculate PCA, "grm" or "randomized"
#     n_components: int
#         Number of components to keep
#     n_power_iter: int
#         Number of power iterations to use for randomized PCA
#     inplace: bool
#         whether to return a new dataset or modify the input dataset
#     """

#     assert method in ["grm", "randomized"], "`method` should be 'grm' or 'randomized'"
#     if method == "grm":
#         if "grm" not in dset.data_vars:
#             # calculate grm
#             if inplace:
#                 grm(dset, inplace=True)
#                 grm_ = dset.data_vars["grm"]
#             else:
#                 grm_ = grm(dset, inplace=False)
#         else:
#             grm_ = dset.data_vars["grm"]
#         # calculate pca
#         u, s, v = da.linalg.svd(grm_)
#         u, s, v = dask.compute(u, s, v)
#         exp_var = (s ** 2) / n_indiv
#         full_var = exp_var.sum()
#         exp_var_ratio = exp_var / full_var

#         coords = u[:, :n_components] * s[:n_components]
#         # TODO: unit test with gcta64
#     elif method == "randomized":
#         # n_indiv, n_snp = gn.shape
#         # if copy:
#         #     gn = gn.copy()

#         # mean_ = gn.mean(axis=0)
#         # std_ = gn.std(axis=0)
#         # gn -= mean_
#         # gn /= std_
#         u, s, v = da.linalg.svd_compressed(
#             dset, n_components=n_components, n_power_iter=n_power_iter
#         )

#         # # calculate explained variance
#         # exp_var = (s ** 2) / n_indiv
#         # full_var = exp_var.sum()
#         # exp_var_ratio = exp_var / full_var

#         # coords = u[:, :n_components] * s[:n_components]

#     # return coords


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
