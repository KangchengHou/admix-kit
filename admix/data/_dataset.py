import pandas as pd
import xarray as xr
import os
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
    dset_eur = xr.open_zarr(join(test_data_path, "eur.zip"))
    dset_afr = xr.open_zarr(join(test_data_path, "afr.zip"))
    dset_admix = xr.open_zarr(join(test_data_path, "admix.zip"))
    return dset_admix, dset_eur, dset_afr

def load_lab_dataset(name: str) -> xr.Dataset:
    """Load prepared dataset in Bogdan lab, currently available
    if you use this function on cluster, and have access to
    `/u/project/pasaniuc/pasaniucdata/admixture/dataset`

    - `simulate_eur_afr.20_80`: Simulated admixture of 20% EUR and 80% AFR
    - `simulate_eur_afr.50_50`: Simulated admixture of 50% EUR and 50% AFR 
    - `ukb_eur_afr`: Admixed individuals in UK Biobank
    
    Returns
    -------
    xr.Dataset
        dataset
    """
    assert name in ["simulate_eur_afr.20_80", "simulate_eur_afr.50_50", "ukb_eur_afr"]
    root_dir = "/u/project/pasaniuc/pasaniucdata/admixture/dataset"
    assert os.path.isdir(root_dir), f"check that you have access to {root_dir}"
        
    dset = xr.open_zarr(join(root_dir, name + ".zip"))
    return dset