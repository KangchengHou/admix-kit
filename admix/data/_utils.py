import numpy as np
import dask.array as da
import pandas as pd
import xarray as xr
import warnings
from pandas.api.types import is_string_dtype
from typing import List, Tuple
import admix


def index_over_chunks(chunks: List[int]):
    """
    iterate over chunks of a dask array

    Parameters
    ----------
    chunks: List[int]
        Number of rows or columns in each chunk
    axis: int
        axis to iterate over 0 for row (SNP by convention), 1 for column (individual by convention)

    Returns
    -------
    generator
        generator of chunk indices so one can access the chunk with
        mat[start : stop, :] (axis=0) or mat[:, start : stop] (axis=1)
    """
    indices = np.insert(np.cumsum(chunks), 0, 0)
    for i in range(len(indices) - 1):
        start, stop = indices[i], indices[i + 1]
        yield start, stop


def match_prs_weights(
    dset: admix.Dataset, df_weight: pd.DataFrame, weight_cols: List[str]
) -> Tuple[admix.Dataset, pd.DataFrame]:
    # TODO: implement this with dapgen.align_snp
    """
    align the dataset and PRS weights with the following 3 steps:
    1. match SNPs in `dset` and `df_weight` by `CHROM` and `POS`
    2. Try match REF and ALT columns in `dset` and `df_weight`, either
        REF_dset = REF_weight, ALT_dset = ALT_weight, or
        REF_dset = ALT_weight, ALT_dset = REF_weight
    3. For mismatched SNPs,all columns in `weight_cols` of returned data frame
        will be reversed

    A shallow copy of `dset` will be returned.
    A copy of `df_weight` will be returned.
    Parameters
    ----------
    dset: a dataset with prs
    df_weight: a dataframe with prs weights
    weight_cols: list of columns in df_weight representing PRS weights
    """

    df_match_dset = (
        dset[["CHROM", "POS", "REF", "ALT"]]
        .to_dataframe()
        .reset_index()
        .rename(columns={"snp": "SNP"})
    )

    df_match_weight = df_weight[["SNP", "CHROM", "POS", "REF", "ALT"]].reset_index()

    df_merged = pd.merge(
        df_match_dset,
        df_match_weight,
        on=["CHROM", "POS"],
        suffixes=["_dset", "_weight"],
    )
    noflip_index = (df_merged["REF_dset"] == df_merged["REF_weight"]) & (
        df_merged["ALT_dset"] == df_merged["ALT_weight"]
    )
    flip_index = (df_merged["REF_dset"] == df_merged["ALT_weight"]) & (
        df_merged["ALT_dset"] == df_merged["REF_weight"]
    )

    rls_df_weight = df_weight.loc[
        df_weight.SNP.isin(df_merged["SNP_weight"][noflip_index | flip_index]),
        ["SNP", "CHROM", "POS", "REF", "ALT"] + weight_cols,
    ].copy()
    rls_df_weight.loc[
        rls_df_weight.SNP.isin(df_merged["SNP_weight"][flip_index]), weight_cols
    ] *= -1

    # the following step can be a bit slow sometimes, need to make sure both array
    # are like <U18 types to be fast
    rls_dset = dset.sel(
        snp=np.isin(
            dset.snp.values,
            df_merged["SNP_dset"][noflip_index | flip_index].values.astype(str),
        )
    )
    return rls_dset, rls_df_weight


def impute_lanc_old(dset: admix.Dataset, dset_ref: admix.Dataset):
    """
    Impute local ancestry using a reference dataset. The two data sets are assumed to
    have the same haplotype order, etc. Typically they are just a subset of each other.

    Using the following steps:
        1. basic checks are performed for the two data sets.
        2. `dset_ref`'s individuals is matched with `dset`, `dset`'s individuals
            must be a subset of `dset_ref`'s individuals.
        3. Imputation are performed

    Parameters
    ----------
    dset: a data set to be imputed with local ancestry
    dset_ref: a data set with local ancestry for reference

    Returns
    -------
    dset_imputed: a data set with imputed local ancestry
    """
    assert (
        len(set(dset.coords["CHROM"].data)) == 1
    ), "Data set to be imputed can only have one chromosome"

    # dset.indiv is a subset of dset_ref.indiv
    assert set(dset.indiv.values) <= set(dset_ref.indiv.values)
    dset_ref = dset_ref.sel(indiv=dset.indiv.values).sel(
        snp=(dset_ref.coords["CHROM"] == dset.coords["CHROM"][0])
    )

    # find relevant regions in reference dataset with local ancestry (hapmap3 SNPs here)
    # ref_start = np.argmin(np.abs(dset_ref["POS"].values - dset["POS"].values[0]))
    ref_start = np.where(dset_ref["POS"] < dset["POS"][0])[0][-1]
    # ref_stop = np.argmin(np.abs(dset_ref["POS"].values - dset["POS"].values[-1]))
    ref_stop = np.where(dset_ref["POS"] > dset["POS"][-1])[0][0]

    dset_ref_subset = dset_ref.isel(snp=slice(ref_start, ref_stop + 1))
    dset_ref_margin = dset_ref_subset.isel(snp=[0, -1])

    imputed_lanc = []
    for ploidy_i in range(2):
        df_ref_margin = pd.DataFrame(
            dset_ref_margin.lanc[:, ploidy_i].values.T,
            columns=dset_ref_margin.indiv.values,
            index=dset_ref_margin.snp.values,
        )
        df_imputed = pd.DataFrame(
            {
                "snp": dset.snp["snp"],
            }
        ).set_index("snp")
        df_imputed = pd.concat(
            [
                df_imputed,
                pd.DataFrame(columns=dset["indiv"].values, dtype=float),
            ]
        )
        # fill margin
        df_imputed = pd.concat(
            [df_ref_margin.iloc[[0], :], df_imputed, df_ref_margin.iloc[[-1], :]],
            axis=0,
        )
        df_imputed.index.name = "snp"
        # fill inside
        df_imputed.loc[
            dset_ref_subset.snp.values, dset_ref_subset.indiv.values
        ] = dset_ref_subset["lanc"][:, :, ploidy_i].values.T
        # interpolate
        df_imputed = (
            df_imputed.reset_index().interpolate(method="nearest").set_index("snp")
        )

        imputed_lanc.append(
            df_imputed.loc[dset["snp"].values, dset["indiv"].values]
            .values.astype(np.int8)
            .T
        )

    dset = dset.assign(
        lanc=(
            ("indiv", "snp", "ploidy"),
            da.from_array(np.dstack(imputed_lanc), chunks=-1),
        )
    )

    return dset


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

            std_row_geno = impute_std(
                geno[:, row_start:row_stop].read().val,
                mean_std["mean"][row_start:row_stop].values,
                mean_std["std"][row_start:row_stop].values,
            )
            std_col_geno = impute_std(
                geno[:, col_start:col_stop].read().val,
                mean_std["mean"][col_start:col_stop].values,
                mean_std["std"][col_start:col_stop].values,
            )

            cov[
                np.ix_(np.arange(row_start, row_stop), np.arange(col_start, col_stop))
            ] = np.dot(std_row_geno.T, std_col_geno) / (std_row_geno.shape[0])
    return cov


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
    data_vars: dict = {
        "geno": (("indiv", "snp", "ploidy"), geno),
    }
    if lanc is not None:
        data_vars["lanc"] = (("indiv", "snp", "ploidy"), lanc)

    coords: dict = {}
    # fill SNP information
    coords["snp"] = snp.index.values
    if not is_string_dtype(coords["snp"]):
        warnings.warn("Transforming snp index to str")
    coords["snp"] = coords["snp"].astype(str)

    for col in snp.columns:
        vals = snp[col].values
        if is_string_dtype(snp[col]):
            vals = snp[col].values.astype(str)

        coords[col] = ("snp", vals)

    # fill in individual information
    coords["indiv"] = indiv.index.values
    if not is_string_dtype(coords["indiv"]):
        warnings.warn("Transforming indiv index to str")
    coords["indiv"] = coords["indiv"].astype(str)

    for col in indiv.columns:
        vals = indiv[col].values
        if is_string_dtype(indiv[col]):
            vals = vals.astype(str)
        coords[col] = ("indiv", vals)

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


def distance_to_line(
    p: np.ndarray, a: np.ndarray, b: np.ndarray, weight: np.ndarray = None
):
    """
    Calculate the distance of each row of a matrix to a reference line

    https://stackoverflow.com/questions/50727961/shortest-distance-between-a-point-and-a-line-in-3-d-space

    Parameters
    ----------
    p: np.ndarray
        matrix (n_indiv, n_var)
    a: np.ndarray
        ref point 1
    b: np.ndarray
        ref point 2
    weight: np.ndarray, optional
        weight associated to each dimension, for example, could be sqrt(eigenvalues)
        if not provided, equal weights for each coordinate will be used

    Returns
    -------
    dist: np.ndarray
        vector distance to the line
    t: np.ndarray
        projected length on the line, normalized according to the length of (b - a)
        t = 0 corresponds projection to a; t = 1 corresponds projection to b.
    n: np.ndarray
        normal vector from the projection point to the original point
        t*b + (1 - t)*a + n can be used to reconstruct the point position.
    """
    if weight is not None:
        p = p * weight
        a = a * weight
        b = b * weight
    x = b - a
    # projected length on the line
    t = np.dot(p - a, x) / np.dot(x, x)
    # orthogonal vector to the line
    n = (p - a) - np.outer(t, x)
    dist = np.linalg.norm(n, axis=1)
    return dist, t, n


def distance_to_refpop(
    sample: np.ndarray, anc1: np.ndarray, anc2: np.ndarray, weight: np.ndarray = None
):
    """
    Calculate the projection distance and position for every sample individuals to the
    line defined by two ancestry populations.

    Parameters
    ----------
    sample: np.ndarray
        (n_indiv, n_dim) matrix
    anc1: np.ndarray
        (n_indiv, n_dim) matrix
    anc2: np.ndarray
        (n_indiv, n_dim) matrix
    weight: np.ndarray, optional
        weight associated to each dimension, for example, could be sqrt(eigenvalues)
        if each dimension corresponds to a principal components.
        if not provided, equal weights for each dimension will be used

    Returns
    -------
    dist: np.ndarray
        normalized distance to the line defined by two ancestral populations
    t: np.ndarray
        projection position
    """
    n_dim = sample.shape[1]
    assert (anc1.shape[1] == n_dim) & (anc2.shape[1] == n_dim)
    if weight is not None:
        assert weight.ndim == 1
        assert len(weight) == n_dim
    p1, p2 = anc1.mean(axis=0), anc2.mean(axis=0)
    sample_dist, sample_t, sample_n = distance_to_line(sample, p1, p2, weight)
    anc1_dist, anc1_t, anc1_n = distance_to_line(anc1, p1, p2, weight)
    anc2_dist, anc2_t, anc2_n = distance_to_line(anc2, p1, p2, weight)
    anc1_maxdist, anc2_maxdist = np.max(anc1_dist), np.max(anc2_dist)
    sample_normalized_dist = sample_dist / (
        (1 - sample_t) * anc1_maxdist + sample_t * anc2_maxdist
    )
    return sample_normalized_dist, sample_t