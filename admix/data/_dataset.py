import pandas as pd
import xarray as xr
from os.path import dirname, join
from typing import List


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
    dset_eur = xr.open_zarr(join(test_data_path, "eur.zarr"))
    dset_afr = xr.open_zarr(join(test_data_path, "afr.zarr"))
    dset_admix = xr.open_zarr(join(test_data_path, "admix.zarr"))
    return dset_admix, dset_eur, dset_afr
