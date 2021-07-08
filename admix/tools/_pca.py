import numpy as np
import pandas as pd
import dask.array as da
import dask


def pca(gn, n_components=10, n_power_iter=4, copy=True):
    """
    gn: (n_indiv, n_snp) matrix
    """
    # standardize to mean 0 and variance 1
    # TODO: check inputs

    n_indiv, n_snp = gn.shape
    if copy:
        gn = gn.copy()

    mean_ = gn.mean(axis=0)
    std_ = gn.std(axis=0)
    gn -= mean_
    gn /= std_

    u, s, v = da.linalg.svd_compressed(gn, k=n_components, n_power_iter=n_power_iter)
    u, s, v = dask.compute(u, s, v)

    # calculate explained variance
    exp_var = (s ** 2) / n_indiv
    full_var = exp_var.sum()
    exp_var_ratio = exp_var / full_var

    coords = u[:, :n_components] * s[:n_components]

    return coords


# def pca(gn, n_components=10, copy=True):
#     # standardize to mean 0 and variance 1
#     # check inputs
#         copy = copy if copy is not None else self.copy
#         gn = asarray_ndim(gn, 2, copy=copy)
#         if not gn.dtype.kind == 'f':
#             gn = gn.astype('f2')

#         # center
#         gn -= self.mean_

#         # scale
#         gn /= self.std_

#     u, s, v = da.linalg.svd_compressed(dset.geno.sum(axis=2).data, k=10, seed=1234)
#     # calculate explained variance
#     self.explained_variance_ = exp_var = (s ** 2) / n_samples
#     full_var = np.var(x, axis=0).sum()
#     self.explained_variance_ratio_ = exp_var / full_var
#             # store components
#     self.components_ = v
#     return u, s, v