# We use `xr.Dataset` as a container for data. We pose some addtional requirements to `xr.Dataset`:
# - It must have `snp` and `indiv` coordinates.
# - Each variable on the `snp` coordinate will have `@snp` as the suffix.
# - Each variable on the `indiv` coordinate will have `@indiv` as the suffix.
# - `n_anc` should appear in the attrs to indicate the number of ancestral populations.

# We have a function `admix.data.check_dataset` to check whether a dataset satisfies these
# requirements.

# Using Xarray and Dask
# ---------------------
# We introduce some commonly used functions and tricks for using Xarray and Dask.

# ```python
# from dask.distributed import Client
# import dask

# client = Client(processes=False, threads_per_worker=1, n_workers=1, memory_limit='20GB')
# ```

# Assign data
# -----------

# Assign a data variable
# ======================
# dset["key"] = (("indiv", "snp", "ploidy"), data))

# Assign a coordinate to the SNP / indiv coordinate
# =================================================
# dset.indiv["key"] = data


# Extract data
# ------------
# dset["key"].data
