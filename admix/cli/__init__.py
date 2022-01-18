#!/usr/bin/env python

import fire
from ._assoc import assoc
from ._utils import log_params
from ._simulate import simulate_pheno
from ._lanc import lanc, lanc_convert, lanc_rfmix
from ._ext import prune, pca


# def merge_dataset(path_list: str, out: str):
#     """Merge multiple dataset [in zarr format] into one dataset, assuming the individiduals
#     are shared typically used for merging multiple datasets from different chromosomes.

#     Parameters
#     ----------
#     path_list : List[str]
#         path of a text file pointing to the list of paths
#     out : str
#         Path to the output zarr file
#     """
#     import xarray as xr

#     dset_list = [xr.open_zarr(p) for p in path_list]

#     dset = xr.concat(dset_list, dim="snp")

#     dset = dset.chunk(chunks={"indiv": -1, "ploidy": -1, "snp": "auto"}).compute()
#     dset.to_zarr(out, mode="w", safe_chunks=False)


def cli():
    """
    Entry point for the admix command line interface.
    """
    fire.Fire()


if __name__ == "__main__":
    fire.Fire()