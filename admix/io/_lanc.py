import numpy as np
from numpy import (
    float32,
    tril_indices_from,
)
import pandas as pd
import dask.array as da
import xarray as xr


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
