import numpy as np
import pandas as pd
from tqdm import tqdm
import dask.array as da
import admix
import dask
from typing import Union, Tuple, List
import dapgen


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


def impute_with_mean(mat, inplace=False, axis=1):
    """impute the each entry using the mean of the input matrix np.mean(mat, axis=axis)
    axis = 1 corresponds to row-wise imputation
    axis = 0 corresponds to column-wise imputation

    Parameters
    ----------
    mat : np.ndarray
        input matrix. For reminder, the genotype matrix is with shape (n_snp, n_indiv)
    inplace : bool
        whether to return a new dataset or modify the input dataset
    axis : int
        axis to impute along

    Returns
    -------
    if inplace:
        mat : np.ndarray
            (n_snp, n_indiv) matrix
    else:
        None
    """
    assert axis in [0, 1], "axis should be 0 or 1"
    if not inplace:
        mat = mat.copy()

    # impute the missing genotypes with the mean of each row
    mean = np.nanmean(mat, axis=axis)
    nanidx = np.where(np.isnan(mat))

    # index the mean using the nanidx[1 - axis]
    # axis = 1, row-wise imputation, index the mean using the nanidx[0]
    # axis = 0, columnw-ise imputation, index the mean using the nanidx[1]
    mat[nanidx] = mean[nanidx[1 - axis]]

    if not inplace:
        return mat
    else:
        return None


def geno_mult_mat(
    geno: da.Array,
    mat: np.ndarray,
    impute_geno: bool = True,
    mat_dim: str = "snp",
    return_snp_var: bool = False,
) -> np.ndarray:
    """Multiply genotype matrix with another matrix

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
        Matrix to be multiplied with the genotype matrix. If the passed variable
        is a vector, it will be transformed to be a 1-column matrix.
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

    if mat.ndim == 1:
        mat = mat[:, np.newaxis]
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


def grm(geno: da.Array, subpopu: np.ndarray = None, std_method: str = "std"):
    """Calculate the GRM matrix
    This function is to serve as an alternative of GCTA --make-grm

    Parameters
    ----------
    geno: admix.Dataset
        genotype (n_snp, n_indiv) matrix
    subpopu : np.ndarray
        subpopulation labels, with shape (n_indiv,). The allele frequencies and
        normalization are performed separately within each subpopulation.
    std_method : str
        Method to standardize the GRM. Currently supported:
        "std" (standardize to have mean 0 and variance 1),
        "allele" (standardize to have mean 0 but no scaling)

    Returns
    -------
    np.ndarray
        GRM matrix (n_indiv, n_indiv)
    """

    def normalize_geno(g):
        """Normalize the genotype matrix"""
        # impute missing genotypes
        g = impute_with_mean(g, inplace=False, axis=1)
        # normalize
        if std_method == "std":
            g = (g - np.mean(g, axis=1)[:, None]) / np.std(g, axis=1)[:, None]
        elif std_method == "allele":
            g = g - np.mean(g, axis=1)[:, None]
        else:
            raise ValueError("std_method should be either `std` or `allele`")
        return g

    assert std_method in ["std", "allele"], "std_method should be `std` or `allele`"
    n_snp = geno.shape[0]
    n_indiv = geno.shape[1]

    if subpopu is not None:
        assert (
            n_indiv == subpopu.shape[0]
        ), "subpopu should have the same length as the number of individuals"
        unique_subpopu = np.unique(subpopu)
        admix.logger.info(
            f"{len(unique_subpopu)} subpopulations found: {unique_subpopu}"
        )

    admix.logger.info(
        f"Calculating GRM matrix with {n_snp} SNPs and {n_indiv} individuals"
    )
    mat = 0
    snp_chunks = geno.chunks[0]
    indices = np.insert(np.cumsum(snp_chunks), 0, 0)
    for i in tqdm(range(len(indices) - 1), desc="admix.data.grm"):
        start, stop = indices[i], indices[i + 1]
        geno_chunk = geno[start:stop, :].compute()
        if subpopu is not None:
            for popu in np.unique(subpopu):
                geno_chunk[:, subpopu == popu] = normalize_geno(
                    geno_chunk[:, subpopu == popu]
                )
        else:
            geno_chunk = normalize_geno(geno_chunk)

        mat += np.dot(geno_chunk.T, geno_chunk) / n_snp
    return mat


def admix_grm(
    geno: da.Array, lanc: da.Array, n_anc: int = 2, snp_prior_var: np.ndarray = None
):
    """Calculate ancestry specific GRM matrix

    Parameters
    ----------
    geno : da.Array
        Genotype matrix with shape (n_snp, n_indiv, 2)
    lanc : np.ndarray
        Local ancestry matrix with shape (n_snp, n_indiv, 2)
    n_anc : int
        Number of ancestral populations
    snp_prior_var : np.ndarray
        Prior variance of each SNP, shape (n_snp,)

    Returns
    -------
    G1: np.ndarray
        ancestry specific GRM matrix for the 1st ancestry
    G2: np.ndarray
        ancestry specific GRM matrix for the 2nd ancestry
    G12: np.ndarray
        ancestry specific GRM matrix for cross term of the 1st and 2nd ancestry
    """

    assert n_anc == 2, "only two-way admixture is implemented"
    assert np.all(geno.shape == lanc.shape)

    apa = admix.data.allele_per_anc(geno, lanc, n_anc=n_anc)
    n_snp, n_indiv = apa.shape[0:2]

    if snp_prior_var is None:
        snp_prior_var = np.ones(n_snp)
    snp_prior_var_sum = snp_prior_var.sum()
    G1 = np.zeros([n_indiv, n_indiv])
    G2 = np.zeros([n_indiv, n_indiv])
    G12 = np.zeros([n_indiv, n_indiv])

    snp_chunks = apa.chunks[0]
    indices = np.insert(np.cumsum(snp_chunks), 0, 0)

    for i in tqdm(range(len(indices) - 1), desc="admix.data.admix_grm"):
        start, stop = indices[i], indices[i + 1]
        apa_chunk = apa[start:stop, :, :].compute()

        # multiply by the prior variance on each SNP
        apa_chunk *= np.sqrt(snp_prior_var[start:stop])[:, None, None]
        a1_chunk, a2_chunk = apa_chunk[:, :, 0], apa_chunk[:, :, 1]

        G1 += np.dot(a1_chunk.T, a1_chunk) / snp_prior_var_sum
        G2 += np.dot(a2_chunk.T, a2_chunk) / snp_prior_var_sum
        G12 += np.dot(a1_chunk.T, a2_chunk) / snp_prior_var_sum

    return G1, G2, G12


def admix_grm_equal_var(
    geno: da.Array, lanc: da.Array, n_anc: int, snp_prior_var: np.ndarray = None
):
    """Calculate ancestry specific GRM matrix K1, K2 (assuming equal variances for ancestries)

    Parameters
    ----------
    geno : da.Array
        Genotype matrix with shape (n_snp, n_indiv, 2)
    lanc : np.ndarray
        Local ancestry matrix with shape (n_snp, n_indiv, 2)
    n_anc : int
        Number of ancestral populations
    snp_prior_var : np.ndarray
        Prior variance of each SNP, shape (n_snp,)

    Returns
    -------
    K1: np.ndarray
        sum of diagonal terms
    K2: np.ndarray
        off-diagonal terms
    """
    assert np.all(geno.shape == lanc.shape)

    apa = admix.data.allele_per_anc(geno, lanc, n_anc=n_anc)
    n_snp, n_indiv = apa.shape[0:2]

    if snp_prior_var is None:
        snp_prior_var = np.ones(n_snp)
    snp_prior_var_sum = snp_prior_var.sum()

    K1 = np.zeros([n_indiv, n_indiv])
    K2 = np.zeros([n_indiv, n_indiv])

    snp_chunks = apa.chunks[0]
    indices = np.insert(np.cumsum(snp_chunks), 0, 0)

    for i in tqdm(range(len(indices) - 1), desc="admix.data.admix_grm_equal_var"):
        start, stop = indices[i], indices[i + 1]
        apa_chunk = apa[start:stop, :, :].compute()

        # multiply by the prior variance on each SNP
        apa_chunk *= np.sqrt(snp_prior_var[start:stop])[:, None, None]

        # diagonal terms
        for i_anc in range(n_anc):
            a_chunk = apa_chunk[:, :, i_anc]
            K1 += np.dot(a_chunk.T, a_chunk) / snp_prior_var_sum

        # off-diagonal terms
        for i_anc in range(n_anc):
            for j_anc in range(i_anc + 1, n_anc):
                a1_chunk, a2_chunk = apa_chunk[:, :, i_anc], apa_chunk[:, :, j_anc]
                K2 += np.dot(a1_chunk.T, a2_chunk) / snp_prior_var_sum

    K2 = K2 + K2.T
    return K1, K2


def admix_ld(dset: admix.Dataset, cov: np.ndarray = None):
    """Calculate ancestry specific LD matrices

    Parameters
    ----------
    dset: admix.Dataset
        dataset containing geno, lanc
    cov : Optional[np.ndarray]
        (n_indiv, n_cov) covariates of the genotypes, an all `1` intercept covariate will always be added
        so there is no need to add the intercept in covariates.
    Returns
    -------
    K1: np.ndarray
        ancestry specific LD matrix for the 1st ancestry
    K2: np.ndarray
        ancestry specific LD matrix for the 2nd ancestry
    K12: np.ndarray
        ancestry specific LD matrix for cross term of the 1st and 2nd ancestry
    """
    assert dset.n_anc == 2, "admix_ld only works for 2 ancestries for now"
    apa = dset.allele_per_anc()

    n_snp, n_indiv = apa.shape[0:2]

    a1, a2 = apa[:, :, 0], apa[:, :, 1]
    if cov is None:
        cov = np.ones((n_indiv, 1))
    else:
        cov = np.hstack([np.ones((n_indiv, 1)), cov])
    # projection = I - X * (X'X)^-1 * X'
    cov_proj_mat = np.eye(n_indiv) - np.linalg.multi_dot(
        [cov, np.linalg.inv(np.dot(cov.T, cov)), cov.T]
    )
    a1 = np.dot(a1, cov_proj_mat)
    a2 = np.dot(a2, cov_proj_mat)
    # center with row mean
    # a1 -= a1.mean(axis=1, keepdims=True)
    # a2 -= a2.mean(axis=1, keepdims=True)
    ld1 = np.dot(a1, a1.T) / n_indiv
    ld2 = np.dot(a2, a2.T) / n_indiv
    ld12 = np.dot(a1, a2.T) / n_indiv
    ld1, ld2, ld12 = dask.compute(ld1, ld2, ld12)
    return {"11": ld1, "22": ld2, "12": ld12}


def af_per_anc(
    geno, lanc, n_anc=2, return_nhaplo=False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Calculate allele frequency per ancestry

    If at one particular SNP locus, no SNP from one particular ancestry can be found
    the corresponding entries will be filled with np.NaN.

    Parameters
    ----------
    geno: np.ndarray
        genotype matrix
    lanc: np.ndarray
        local ancestry matrix
    n_anc: int
        number of ancestries
    return_nhaplo: bool
        whether to return the number of haplotypes per ancestry

    Returns
    -------
    np.ndarray
        (n_snp, n_anc) length list of allele frequencies.
    """
    assert np.all(geno.shape == lanc.shape)
    n_snp = geno.shape[0]
    af = np.zeros((n_snp, n_anc))
    lanc_nhaplo = np.zeros((n_snp, n_anc))
    snp_chunks = geno.chunks[0]
    indices = np.insert(np.cumsum(snp_chunks), 0, 0)

    for i in tqdm(range(len(indices) - 1), desc="admix.data.af_per_anc"):
        start, stop = indices[i], indices[i + 1]
        geno_chunk = geno[start:stop, :, :].compute()
        lanc_chunk = lanc[start:stop, :, :].compute()

        for anc_i in range(n_anc):
            lanc_mask = lanc_chunk == anc_i
            lanc_nhaplo[start:stop, anc_i] = np.sum(lanc_mask, axis=(1, 2))
            # mask SNPs with local ancestry not `i_anc`
            af[start:stop, anc_i] = (
                np.ma.masked_where(np.logical_not(lanc_mask), geno_chunk)
                .sum(axis=(1, 2))
                .data
            ) / lanc_nhaplo[start:stop, anc_i]

    if return_nhaplo:
        return af, lanc_nhaplo
    else:
        return af


def allele_per_anc(
    geno: da.Array,
    lanc: da.Array,
    n_anc: int,
    center=False,
):
    """Get allele count per ancestry

    Parameters
    ----------
    geno: da.Array
        genotype data
    lanc: da.Array
        local ancestry data
    n_anc: int
        number of ancestries

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

    assert (
        geno.chunks == lanc.chunks
    ), "`geno` and `lanc` should have the same chunk size"

    assert len(geno.chunks[1]) == 1, (
        "geno / lanc should not be chunked across the second dimension"
        "(individual dimension)"
    )

    def helper(geno_chunk, lanc_chunk, n_anc):
        n_snp, n_indiv, n_haplo = geno_chunk.shape
        apa = np.zeros((n_snp, n_indiv, n_anc), dtype=np.float64)
        for i_haplo in range(n_haplo):
            haplo_hap = geno_chunk[:, :, i_haplo]
            haplo_lanc = lanc_chunk[:, :, i_haplo]
            for i_anc in range(n_anc):
                apa[:, :, i_anc][haplo_lanc == i_anc] += haplo_hap[haplo_lanc == i_anc]
        return apa

    # the resulting chunk sizes will be the same as the input for snp, indiv
    # while the third dimension will be (n_anc, )
    output_chunks = (geno.chunks[0], geno.chunks[1], (n_anc,))
    res = da.map_blocks(
        lambda geno_chunk, lanc_chunk: helper(
            geno_chunk=geno_chunk, lanc_chunk=lanc_chunk, n_anc=n_anc
        ),
        geno,
        lanc,
        dtype=np.float64,
        chunks=output_chunks,
    )

    return res


def calc_pgs(dset: admix.Dataset, df_weights: pd.DataFrame, method: str):
    """Calculate PGS for each individual

    Parameters
    ----------
    dset: admix.Dataset
        dataset object
    df_weights: pd.DataFrame
        weights for each individual
    method: str
        method to calculate PGS. Options are:
        - "total": vanilla PGS
        - "partial": partial PGS, calculate partial PGS for each local ancestry

    Returns
    -------
    np.ndarray
        PGS for each individual
        - method = "total": (n_indiv, )
        - method = "partial": (n_indiv, n_anc)
    """
    assert method in [
        "total",
        "partial",
    ], "method should be either 'total' or 'partial'"
    assert np.all(
        dset.snp.index == df_weights.index
    ), "`dset` and `df_weights` should have exactly the same index"

    assert len(df_weights.columns) == 1, "`df_weights` should have only one column"

    if method == "total":
        pgs = admix.data.geno_mult_mat(
            dset.geno.sum(axis=2), df_weights.values
        ).flatten()
    elif method == "partial":
        n_anc = dset.n_anc
        pgs = np.zeros((dset.n_indiv, n_anc))
        apa = dset.allele_per_anc()
        for i_anc in range(n_anc):
            pgs[:, i_anc] = admix.data.geno_mult_mat(
                apa[:, :, i_anc], df_weights.values
            ).flatten()

    else:
        raise ValueError("method should be either 'total' or 'partial'")

    return pgs


def calc_partial_pgs(
    dset: admix.Dataset,
    df_weights: pd.DataFrame,
    dset_ref: admix.Dataset = None,
    ref_pop_indiv: List[List[str]] = None,
    weight_col="WEIGHT",
) -> pd.DataFrame:
    """Calculate PGS for each individual

    Parameters
    ----------
    dset: admix.Dataset
        dataset object
    df_weights: pd.DataFrame
        weights for each individual
    dset_ref: admix.Dataset
        reference dataset object, use `dapgen.align_snp` to align the SNPs between
        `dset` and `dset_ref`. `CHROM` and `POS` must match, with potential flips of
        `REF` and `ALT` allele coding.
    ref_pop: List[List[str]]
        list of reference individual ID in `dset_ref`

    Returns
    -------
    pd.DataFrame
        PGS for each individual
        - (n_indiv, n_anc)
    """
    assert (dset_ref is None) == (
        ref_pop_indiv is None
    ), "both `dset_ref` and `ref_pop_indiv` should be None or not None"
    CALC_REF = dset_ref is not None
    CHECK_COLS = ["CHROM", "POS", "REF", "ALT"]
    ## check input
    idx1, idx2, sample_wgt_flip = dapgen.align_snp(
        df1=dset.snp[CHECK_COLS], df2=df_weights[CHECK_COLS]
    )

    assert np.all(idx1 == dset.snp.index) & np.all(
        idx2 == df_weights.index
    ), "`dset` and `df_weights` should align, with potential allele flip"

    if CALC_REF:
        idx1, idx2, ref_wgt_flip = dapgen.align_snp(
            df1=dset.snp[CHECK_COLS], df2=dset_ref.snp[CHECK_COLS]
        )
        assert np.all(idx1 == dset.snp.index) & np.all(
            idx2 == dset_ref.snp.index
        ), "`dset` and `dset_ref` should align, with potential allele flip"

    weights = df_weights[weight_col].values
    sample_weights = weights * sample_wgt_flip

    if CALC_REF:
        ref_weights = weights * ref_wgt_flip * sample_wgt_flip

        assert (
            len(ref_pop_indiv) == dset.n_anc
        ), "`len(ref_pops)` should match with `dset.n_anc`"

    ## scoring
    dset_geno, dset_lanc = dset.geno.compute(), dset.lanc.compute()
    sample_pgs = np.zeros((dset.n_indiv, dset.n_anc))
    if CALC_REF:
        ref_geno_list = [dset_ref[:, pop].geno.compute() for pop in ref_pop_indiv]
        ref_pgs = [[] for pop in ref_pop_indiv]
    # iterate over each individuals
    for indiv_i in tqdm(range(dset.n_indiv), desc="admix.data.calc_partial_pgs"):
        indiv_ref_pgs = [0, 0]
        # pgs for sample individuals
        for haplo_i in range(2):
            geno = dset_geno[:, indiv_i, haplo_i]
            lanc = dset_lanc[:, indiv_i, haplo_i]
            for lanc_i in range(dset.n_anc):
                # sample
                sample_pgs[indiv_i, lanc_i] += np.dot(
                    geno[lanc == lanc_i], sample_weights[lanc == lanc_i]
                )

                # pgs for reference individuals
                if CALC_REF:
                    ref_geno = ref_geno_list[lanc_i][lanc == lanc_i, :, :]
                    if ref_geno.shape[0] > 0:
                        ref_geno = ref_geno.reshape(ref_geno.shape[0], -1)
                        s = np.dot(ref_weights[lanc == lanc_i], ref_geno)
                    else:
                        s = np.zeros(ref_geno.shape[1] * 2)
                    indiv_ref_pgs[lanc_i] += s
        if CALC_REF:
            for lanc_i in range(dset.n_anc):
                ref_pgs[lanc_i].append(indiv_ref_pgs[lanc_i])

    # format ref_pgs: for each ancestry, we have n_indiv x (n_ref_indiv x 2)
    # each reference has 2 haplotypes
    if CALC_REF:
        ref_pgs = [
            pd.DataFrame(
                data=np.vstack(ref_pgs[i]),
                index=dset.indiv.index,
                columns=np.concatenate(
                    [[str(i) + "_1", str(i) + "_2"] for i in ref_pop_indiv[i]]
                ),
            )
            for i in range(dset.n_anc)
        ]
    sample_pgs = pd.DataFrame(
        data=sample_pgs,
        index=dset.indiv.index,
        columns=[f"ANC{i}" for i in range(1, dset.n_anc + 1)],
    )
    if CALC_REF:
        return sample_pgs, ref_pgs
    else:
        return sample_pgs