import numpy as np
import pandas as pd
# import allel

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