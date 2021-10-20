"""
Sept. 24, 2021: this file is not in any use. After other modules are mature enough,
    we can devote effort to design a Dataset class.
"""

import pandas as pd
import xarray as xr
import os
import numpy as np
from pandas.api.types import infer_dtype, is_string_dtype, is_categorical_dtype
import warnings
import dask.array as da
from typing import (
    Hashable,
    List,
    Optional,
    Tuple,
    Union,
    Mapping,
    Any,
    Dict,
    Sequence,
    Iterable,
    Literal,
    Mapping,
    MutableMapping,
)

from xarray.core.dataset import DataVariables


class Dataset(object):
    """
    Class to handle the dataset.

    An admix.Dataset `dset` Support the following operations:

    - dset.indiv =

    Design principles
    -----------------
    Use xr.Dataset to take charge of `geno` and `lanc` and arrays with >= 2 dimensions
    such as `af_per_anc`, `allele_per_anc`

    Use pd.Dataframe to represent `snp` and `indiv`

    References
    ----------
    scanpy

    """

    def __init__(
        self,
        geno: Optional[da.Array] = None,
        lanc: Optional[da.Array] = None,
        indiv: Optional[pd.DataFrame] = None,
        snp: Optional[pd.DataFrame] = None,
        uns: Optional[Mapping[str, Any]] = None,
        n_anc: int = None,
    ):

        data_vars: Dict[Hashable, Any] = {}

        data_vars["geno"] = (("snp", "indiv", "ploidy"), geno)
        if lanc is not None:
            data_vars["lanc"] = (("snp", "indiv", "ploidy"), lanc)

        self._snp = snp
        self._indiv = indiv
        # coords: Dict[Hashable, Any] = {}
        # if snp is not None:

        #     # fill SNP information
        #     coords["snp"] = snp.index.values
        #     if not is_string_dtype(coords["snp"]):
        #         warnings.warn("Transforming snp index to str")
        #     coords["snp"] = coords["snp"].astype(str)

        #     for col in snp.columns:
        #         vals = snp[col].values
        #         if is_string_dtype(snp[col]):
        #             vals = snp[col].values.astype(str)

        #         coords[f"{col}@snp"] = ("snp", vals)

        self._xr = xr.Dataset(data_vars=data_vars)

        if ("lanc" in self._xr) and (n_anc is None):
            # infer number of ancestors
            n_anc = int(lanc.max().compute() + 1)
            self._xr.attrs["n_anc"] = n_anc
        else:
            self._xr.attrs["n_anc"] = n_anc
        # self._check_dimensions()
        # self._check_uniqueness()

    def _check_dimensions(self) -> None:
        """
        Check that all dimensions are unique.
        """
        pass

    def _check_uniqueness(self) -> None:
        """
        Check that all dimensions are unique.
        """
        pass

    def __repr__(self) -> str:
        descr = f"admix.Dataset object with n_indiv={self.n_indiv}, n_snp={self.n_snp}"
        if "lanc" in self._xr:
            descr += f", n_anc={self.n_anc}"
        else:
            descr += " with no local ancestries"
        return descr

    def update(self, other, dim="data") -> "Dataset":
        """
        Update the dataset with the information from another dataset.
        This is an inplace operation.

        Parameters
        ----------
        var : one of the following:
            - mapping {var name: (dimension name, array-like)} for "data"
            - array-like for "indiv"
            - array-like for "snp"

        dim: str
            The dimension to use for the update.
            Default: "data" (i.e. the data variable)
            one of the following:
            - "data": data variables
            - "indiv": individual variables
            - "snp": SNP variables

        Returns
        -------
        self : admix.Dataset
            Updated Dataset.
        """
        if dim == "data":
            # TODO: check that the dimensions are the same
            self._xr.update(other)
        elif dim == "indiv":
            self._xr.update({f"{key}@indiv": val for key, val in other.items()})
        elif dim == "snp":
            self._xr.update({f"{key}@snp": val for key, val in other.items()})
        else:
            raise ValueError(f"Invalid dimension: {dim}")

        return self

    @property
    def n_indiv(self) -> int:
        """Number of individuals."""
        return self._xr.dims["indiv"]

    @property
    def n_snp(self) -> int:
        """Number of SNPs."""
        return self._xr.dims["snp"]

    @property
    def n_anc(self) -> int:
        """Number of ancestries."""
        return self._xr.attrs["n_anc"]

    @property
    def data(self) -> DataVariables:
        """Number of individuals."""
        return self._xr.data_vars

    @property
    def indiv(self) -> pd.DataFrame:
        """One-dimensional annotation of observations (`pd.DataFrame`)."""
        return self._xr

    @property
    def uns(self) -> MutableMapping:
        """Unstructured annotation (ordered dictionary)."""
        uns = self._xr.attrs.get("uns", None)
        return uns

    def __getitem__(self, index) -> "Dataset":
        """Returns a sliced view of the object."""
        snp_idx, indiv_idx = index
        # snp_idx, indiv_idx = self._normalize_indices(index)
        return Dataset(geno, snp_idx=snp_idx, indiv_idx=indiv_idx)