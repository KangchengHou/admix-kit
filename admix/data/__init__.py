"""
all about admix.Dataset
"""
import xarray as xr
from ._utils import (
    make_dataset,
    load_toy,
    load_lab_dataset,
    quantile_normalize,
    impute_lanc,
    match_prs_weights,
)


def assign_lanc(dset: xr.Dataset, lanc_file: str, format: str = "rfmix"):
    """
    Assign local ancestry to a dataset. Currently we assume that the rfmix file contains
    2-way admixture information.

    Parameters
    ----------
    dset: xr.Dataset
        Dataset to assign local ancestry to.
    lanc_file: str
        Path to local ancestry data.
    format: str
        Format of local ancestry data.
        Currently only "rfmix" is supported.

    Returns
    -------
    dset: xr.Dataset
        Dataset with local ancestry assigned.
    TODO:
    - Add support for other formats.
    """
    import pandas as pd
    import numpy as np

    assert format in ["rfmix"], "Only rfmix format is supported."
    # assign local ancestry
    rfmix = pd.read_csv(lanc_file, sep="\t", skiprows=1)
    lanc_full = np.full(
        (dset.dims["indiv"], dset.dims["snp"], dset.dims["ploidy"]), np.nan
    )
    lanc0 = rfmix.loc[:, rfmix.columns.str.endswith(".0")].rename(
        columns=lambda x: x[:-2]
    )
    lanc1 = rfmix.loc[:, rfmix.columns.str.endswith(".1")].rename(
        columns=lambda x: x[:-2]
    )

    assert np.all(dset.indiv == lanc0.columns)
    for i_row, row in rfmix.iterrows():
        mask_row = (
            (row.spos <= dset.snp["POS"]) & (dset.snp["POS"] <= row.epos)
        ).values
        lanc_full[:, mask_row, 0] = lanc0.loc[i_row, :].values[:, np.newaxis]
        lanc_full[:, mask_row, 1] = lanc1.loc[i_row, :].values[:, np.newaxis]
    lanc_full = lanc_full.astype(np.int8)
    dset = dset.assign({"lanc": (("indiv", "snp", "ploidy"), lanc_full)})
    dset = dset.assign_attrs({"n_anc": 2})
    return dset


__all__ = [
    "read_digit_mat",
    "write_digit_mat",
    "compute_allele_per_anc",
    "load_toy",
    "load_lab_dataset",
    "make_dataset",
    "assign_lanc",
]
