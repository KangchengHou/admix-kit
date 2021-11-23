import numpy as np
from numpy import (
    float32,
    tril_indices_from,
)
import pandas as pd
import dask
import dask.array as da
import xarray as xr
from tqdm import tqdm
from dask.array import concatenate, from_delayed
from dask.delayed import delayed
from bisect import bisect_left
from typing import List


def check_lanc(break_list: List[List[int]], value_list: List[List[str]]):
    """Check the format of .lanc file

    Parameters
    ----------
    break_list : List[List[int]]
        List of break points
    value_list : List[List[str]]
        List of value

    Returns
    -------
    Tuple[int, int]
        (n_snp, n_indiv)
    """
    assert len(break_list) == len(
        value_list
    ), "`break_list` and `value_list` must have the same length (same as n_indiv)"

    assert np.all([len(b) == len(v) for b, v in zip(break_list, value_list)])
    n_snp = break_list[0][-1]
    n_indiv = len(break_list)
    assert np.all(
        n_snp == b[-1] for b in break_list
    ), "The last element of break points for each individual in `break_list` must be equal"
    return n_snp, n_indiv


def _subset_lanc(start: int, stop: int, break_list, value_list):
    """
    Subset the .lanc file

    Parameters
    ----------
    start : int
        start of SNP
    stop : int
        stop of SNP
    break_list: List[List[int]]
        list of break points
    value_list: List[List[str]]

    """
    # For each individual, find index of break points that's within [start, stop]
    start_idx = [bisect_left(indiv_break, start) for indiv_break in break_list]
    stop_idx = [bisect_left(indiv_break, stop) for indiv_break in break_list]
    new_break_list = [
        br[s:e] + [stop] for s, e, br in zip(start_idx, stop_idx, break_list)
    ]
    # offset with start
    new_break_list = [[b - start for b in br] for br in new_break_list]
    # find corresponding value
    new_value_list = [
        val[s:e] + [val[e]] for s, e, val in zip(start_idx, stop_idx, value_list)
    ]

    return new_break_list, new_value_list


def lanc_to_numpy(break_list, value_list):
    """
    Given `break_list` list of break points with `n_indiv` length
    And the corresponding `value_list`, the correponding value

    Convert to dense matrix
    """
    n_indiv = len(break_list)
    n_snp = break_list[0][-1]
    mat = np.full((n_snp, n_indiv, 2), -1)
    for indiv_i in range(n_indiv):
        start = 0
        for stop, val in zip(break_list[indiv_i], value_list[indiv_i]):
            a1, a2 = int(val[0]), int(val[1])
            mat[start:stop, indiv_i, 0] = a1
            mat[start:stop, indiv_i, 1] = a2
            start = stop
    return mat


def lanc_to_dask(break_list, value_list, snp_chunk: int):
    """
    Given `break_list` list of break points with `n_indiv` length
    And the corresponding `value_list`, the correponding value

    Convert to dask matrix
    """
    n_indiv = len(break_list)
    n_snp = break_list[0][-1]

    n_snp = break_list[0][-1]
    assert np.all(
        n_snp == b[-1] for b in break_list
    ), "The last element of break points for each individual in `break_list` must be equal"

    # all local ancestries
    lancs = []

    read_subset_lanc = lambda start, stop: lanc_to_numpy(
        *_subset_lanc(start, stop, break_list, value_list)
    )
    snp_start = 0
    while snp_start < n_snp:
        snp_stop = min(snp_start + snp_chunk, n_snp)
        shape = (snp_stop - snp_start, n_indiv, 2)

        lancs.append(
            from_delayed(
                value=delayed(read_subset_lanc)(snp_start, snp_stop),
                shape=shape,
                dtype=np.int8,
            )
        )
        snp_start = snp_stop
    return concatenate(lancs, 0, False)


def read_lanc(path: str, snp_chunk: int = 1024, return_dask=True) -> da.Array:
    """Read local ancestry with .lanc format

    Parameters
    ----------
    path : .lanc file
        Local ancestry file

    Returns
    -------
    da.Array
        (n_snp, n_indiv, 2)
    """

    # TODO: first check input
    # the end must equal to n_snp
    with open(path) as f:
        lines = f.readlines()
    n_snp, n_indiv = [int(i) for i in lines[0].strip().split()]
    data_list = [line.strip().split() for line in lines[1:]]
    assert len(data_list) == n_indiv
    break_list = [[int(l.split(":")[0]) for l in line] for line in data_list]
    value_list = [[l.split(":")[1] for l in line] for line in data_list]

    if return_dask is True:
        return lanc_to_dask(break_list, value_list, snp_chunk)
    else:
        return break_list, value_list


def read_lanc_slow(path: str) -> da.Array:
    """Read local ancestry with .lanc format

    Parameters
    ----------
    path : .lanc file
        Local ancestry file

    Returns
    -------
    da.Array
        (n_snp, n_indiv, 2)
    """

    # TODO: first check input
    # the end must equal to n_snp
    with open(path) as f:
        lines = f.readlines()
    n_snp, n_indiv = [int(i) for i in lines[0].strip().split()]
    lanc_list = [line.strip().split() for line in lines[1:]]
    assert len(lanc_list) == n_indiv

    lanc_mat = da.zeros((n_snp, n_indiv, 2), dtype=np.int8)
    for indiv_i, indiv_lanc in tqdm(enumerate(lanc_list)):
        start = 0
        for l in indiv_lanc:
            a = l.split(":")[1]
            a1, a2 = int(a[0]), int(a[1])
            stop = int(l.split(":")[0])
            lanc_mat[start:stop, indiv_i, 0] = a1
            lanc_mat[start:stop, indiv_i, 1] = a2
            start = stop
    return lanc_mat


def write_lanc(
    path: str, lanc: da.Array = None, break_list: List = None, value_list: List = None
):

    if lanc is not None:
        assert (break_list is None) and (
            value_list is None
        ), "when `lanc` is specified, `break_list` and `value_list` must both be None"

        n_snp, n_indiv, n_ploidy = lanc.shape
        assert n_ploidy == 2, "`lanc` must be (n_snp, n_indiv, 2)"

        # convert to dask.array if numpy array
        if isinstance(lanc, np.ndarray):
            lanc = da.from_array(lanc)

        assert isinstance(lanc, da.Array), "`lanc` must be dask array"

        # switch points
        snp_pos, indiv_pos, _, = dask.compute(
            np.where(lanc[1:, :, :] != lanc[0:-1, :, :]), scheduler="single-threaded"
        )[0]
        # end points for all the individuals
        snp_pos = np.concatenate([snp_pos, [n_snp - 1] * n_indiv])
        indiv_pos = np.concatenate([indiv_pos, np.arange(n_indiv)])
        # (snp, indiv) -> snp * n_indiv + indiv
        values = lanc.reshape([-1, 2])[indiv_pos + snp_pos * n_indiv, :].compute()
        values = np.array([str(v[0]) + str(v[1]) for v in values])

        break_list = []
        value_list = []
        for indiv_i in range(n_indiv):
            indiv_mask = indiv_pos == indiv_i
            # +1 because .lanc denote the [start, stop) right-open interval
            indiv_snp_pos, unique_mask = np.unique(
                snp_pos[indiv_mask] + 1, return_index=True
            )
            indiv_values = values[indiv_mask][unique_mask]
            break_list.append(indiv_snp_pos.tolist())
            value_list.append(indiv_values.tolist())
    else:
        assert (break_list is not None) and (
            value_list is not None
        ), "when `lanc` is None, `break_list` and `value_list` must both be specified"
        n_snp, n_indiv = check_lanc(break_list, value_list)

    # write to file
    lines = []

    # header
    lines.append(f"{n_snp} {n_indiv}")

    for indiv_break, indiv_value in zip(break_list, value_list):
        lines.append(
            " ".join([str(b) + ":" + v for (b, v) in zip(indiv_break, indiv_value)])
        )

    with open(path, "w") as f:
        f.writelines("\n".join(lines))


def write_lanc_slow(path: str, lanc: da.Array):
    """Write local ancestry `lanc` to `path`

    Parameters
    ----------
    path : str
        Path of local ancestry
    lanc : da.Array
        (n_snp, n_indiv, 2) local ancestry matrix
    """

    n_snp, n_indiv, _ = lanc.shape
    # find break points
    snp_pos, indiv_pos, _ = dask.compute(
        np.where(lanc[1:, :, :] != lanc[0:-1, :, :]), scheduler="single-threaded"
    )[0]

    f = open(path, "w")
    # write header
    f.writelines(f"{n_snp} {n_indiv}\n")

    lines = []
    for indiv_i in range(n_indiv):
        indiv_snp_pos = (snp_pos[indiv_pos == indiv_i] + 1).tolist() + [lanc.shape[0]]
        indiv_snp_pos = np.sort(np.unique(indiv_snp_pos))
        lines.append(
            " ".join(
                [
                    str(lanc[s - 1, indiv_i, 0].compute())
                    + str(lanc[s - 1, indiv_i, 1].compute())
                    + ":"
                    + str(s)
                    for s in indiv_snp_pos
                ]
            )
        )
    f.writelines("\n".join(lines))
    f.close()


def read_rfmix(
    path: str,
    geno: xr.DataArray,
    df_snp: pd.DataFrame,
    return_dask: bool = True,
    snp_chunk: int = 1024,
) -> da.Array:
    """
    Assign local ancestry to a dataset. Currently we assume that the rfmix file contains
    2-way admixture information.

    # TODO: directly from rfmix to .lanc format

    Parameters
    ----------
    lanc_file: str
        Path to local ancestry data.
    geno: xr.DataArray
        genotype matrix
    df_snp: pd.DataFrame
        SNP data frames

    Returns
    -------
    lanc: da.Array
        Local ancestry array
    """

    # assign local ancestry
    df_rfmix = pd.read_csv(path, sep="\t", skiprows=1)

    assert np.all(geno.snp == df_snp.index), "geno.snp should match with `df_snp`"

    # TODO: currently assume 2-way admixture
    # MORE THAN 2-way admixture is easily supported by reading the header and modify
    # the following 6 lines
    lanc0 = df_rfmix.loc[:, df_rfmix.columns.str.endswith(".0")].rename(
        columns=lambda x: x[:-2]
    )
    lanc1 = df_rfmix.loc[:, df_rfmix.columns.str.endswith(".1")].rename(
        columns=lambda x: x[:-2]
    )

    lanc = lanc0.astype(str) + lanc1.astype(str)

    df_rfmix_info = df_rfmix.iloc[:, 0:3].copy()
    # extend local ancestry to two ends of chromosomes
    df_rfmix_info.loc[0, "spos"] = df_snp["POS"][0] - 1
    df_rfmix_info.loc[len(df_rfmix_info) - 1, "epos"] = df_snp["POS"][-1] + 1

    assert np.all(geno.indiv == lanc.columns)

    n_indiv = geno.sizes["indiv"]
    n_snp = geno.sizes["snp"]

    rfmix_break_list = []
    # [start, stop) of SNPs for each rfmix break points
    for chunk_i, chunk in df_rfmix_info.iterrows():
        chunk_mask = (
            (chunk.spos <= df_snp["POS"]) & (df_snp["POS"] < chunk.epos)
        ).values
        chunk_start, chunk_stop = np.where(chunk_mask)[0][[0, -1]]
        rfmix_break_list.append(chunk_stop)
    rfmix_break_list = np.array(rfmix_break_list)

    # find break points in the data
    chunk_pos, indiv_pos = np.where(lanc.iloc[1:, :].values != lanc.iloc[:-1, :].values)
    # convert to SNP positions
    snp_pos = rfmix_break_list[chunk_pos]
    values = lanc.values[chunk_pos, indiv_pos]

    # append values at the end of the chromosomes
    snp_pos = np.concatenate([snp_pos, [n_snp - 1] * n_indiv])
    indiv_pos = np.concatenate([indiv_pos, np.arange(n_indiv)])
    values = np.concatenate([values, lanc.iloc[-1].values])

    # snp_pos, indiv_pos, values are now triples of break points

    break_list = []
    value_list = []
    # convert to .lanc format
    for indiv_i in range(n_indiv):
        indiv_mask = indiv_pos == indiv_i
        # +1 because .lanc denote the [start, stop) of the break points
        indiv_snp_pos, unique_mask = np.unique(
            snp_pos[indiv_mask] + 1, return_index=True
        )
        indiv_values = values[indiv_mask][unique_mask]
        break_list.append(indiv_snp_pos.tolist())
        value_list.append(indiv_values.tolist())

    if return_dask:
        return lanc_to_dask(break_list, value_list, snp_chunk=snp_chunk)
    else:
        return break_list, value_list


def read_rfmix_deprecated(
    lanc_file: str, geno: xr.DataArray, df_snp: pd.DataFrame
) -> da.Array:
    """
    Assign local ancestry to a dataset. Currently we assume that the rfmix file contains
    2-way admixture information.

    # TODO: directly from rfmix to .lanc format

    Parameters
    ----------
    lanc_file: str
        Path to local ancestry data.
    geno: xr.DataArray
        genotype matrix
    df_snp: pd.DataFrame
        SNP data frames

    Returns
    -------
    lanc: da.Array
        Local ancestry array
    """

    # assign local ancestry
    rfmix = pd.read_csv(lanc_file, sep="\t", skiprows=1)

    lanc_full = da.full(
        shape=(geno.sizes["indiv"], geno.sizes["snp"], geno.sizes["ploidy"]),
        fill_value=-1,
        dtype=np.int8,
    )
    assert np.all(geno.snp == df_snp.index), "geno.snp should match with `df_snp`"

    # TODO: currently assume 2-way admixture
    # MORE THAN 2-way admixture is easily supported by reading the header and modify
    # the following 6 lines
    lanc0 = rfmix.loc[:, rfmix.columns.str.endswith(".0")].rename(
        columns=lambda x: x[:-2]
    )
    lanc1 = rfmix.loc[:, rfmix.columns.str.endswith(".1")].rename(
        columns=lambda x: x[:-2]
    )

    assert np.all(geno.indiv == lanc0.columns)
    assert np.all(geno.indiv == lanc1.columns)

    for i_row, row in rfmix.iterrows():
        mask_row = ((row.spos <= df_snp["POS"]) & (df_snp["POS"] <= row.epos)).values
        lanc_full[:, mask_row, 0] = lanc0.loc[i_row, :].values[:, np.newaxis]
        lanc_full[:, mask_row, 1] = lanc1.loc[i_row, :].values[:, np.newaxis]

    dset_names = tuple(d for d in geno.sizes)
    if dset_names == ("indiv", "snp", "ploidy"):
        # do nothing
        pass
    elif dset_names == ("snp", "indiv", "ploidy"):
        lanc_full = lanc_full.swapaxes(0, 1)
    else:
        raise ValueError(
            f"Unexpected dimensions {dset_names}. "
            "Expected (indiv, snp, ploidy) or (snp, indiv, ploidy)"
        )

    lanc_full = lanc_full.rechunk(geno.chunks)

    return lanc_full