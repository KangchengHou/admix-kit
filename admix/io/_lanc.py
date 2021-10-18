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


def _lanc_to_dense(break_list, value_list):
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

def read_lanc(path: str, snp_chunk: int = 1024) -> da.Array:
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
    
    # all local ancestries
    lancs = []
    
    read_subset_lanc = lambda start, stop : _lanc_to_dense(*_subset_lanc(start, stop, break_list, value_list))
    snp_start = 0
    while snp_start < n_snp:
        snp_stop = min(snp_start + snp_chunk, n_snp)
        shape = (snp_stop - snp_start, n_indiv, 2)

        lancs.append(
            from_delayed(
                value=delayed(read_subset_lanc)(
                    snp_start, snp_stop
                ),
                shape=shape,
                dtype=np.int8,
            )
        )
        snp_start = snp_stop
    return concatenate(lancs, 0, False)

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

def write_lanc(path: str, lanc: da.Array):
    n_snp, n_indiv, _ = lanc.shape

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

    lines = []

    # header
    lines.append(f"{n_snp} {n_indiv}")

    for indiv_i in range(n_indiv):
        indiv_mask = indiv_pos == indiv_i
        # +1 because .lanc denote the starting point
        indiv_snp_pos, unique_mask = np.unique(
            snp_pos[indiv_mask] + 1, return_index=True
        )
        indiv_values = values[indiv_mask][unique_mask]

        lines.append(
            " ".join([str(p) + ":" + v for (p, v) in zip(indiv_snp_pos, indiv_values)])
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
    snp_pos, indiv_pos, _ = dask.compute(np.where(lanc[1:, :, :] != lanc[0:-1, :, :]), 
        scheduler='single-threaded')[0]

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


def read_rfmix(lanc_file: str, geno: xr.DataArray, df_snp: pd.DataFrame) -> da.Array:
    """
    Assign local ancestry to a dataset. Currently we assume that the rfmix file contains
    2-way admixture information.

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