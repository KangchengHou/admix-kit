import pandas as pd
import xarray as xr
import numpy as np
import dask.array as da
from xarray.core.dataset import DataVariables
import admix
import dask
from typing import (
    Hashable,
    List,
    Optional,
    Mapping,
    Any,
    Dict,
    Mapping,
    MutableMapping,
)
from ._utils import normalize_indices


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
        snp: Optional[pd.DataFrame] = None,
        indiv: Optional[pd.DataFrame] = None,
        n_anc: int = None,
        dset_ref=None,
        snp_idx=None,
        indiv_idx=None,
    ):
        if dset_ref is not None:
            # initialize from reference data set
            if isinstance(snp_idx, (int, np.integer)):
                assert (
                    0 <= snp_idx < dset_ref.n_snp
                ), "SNP index `{snp_idx}` is out of range."
                snp_idx = slice(snp_idx, snp_idx + 1, 1)

            if isinstance(indiv_idx, (int, np.integer)):
                assert (
                    0 <= indiv_idx < dset_ref.n_indiv
                ), "SNP index `{indiv_idx}` is out of range."
                indiv_idx = slice(indiv_idx, indiv_idx + 1, 1)

            # make sure `snp_idx` and `indiv_idx` are in sorted order according to the
            # original index

            # TODO: the following may be a performance bottleneck, optimize this if needed
            # TODO: rather than raise an error, we could just adjust the ordering.
            subset_indiv = dset_ref.indiv.iloc[indiv_idx, :].copy()

            assert np.all(
                dset_ref.indiv.index[
                    dset_ref.indiv.index.isin(subset_indiv.index.values)
                ]
                == subset_indiv.index.values
            ), "the index provided must be in the same order as in the original order"

            subset_snp = dset_ref.snp.iloc[snp_idx, :].copy()

            assert np.all(
                dset_ref.snp.index[dset_ref.snp.index.isin(subset_snp.index.values)]
                == subset_snp.index.values
            ), "the index provided must be in the same order as in the original order"

            self._indiv = subset_indiv
            self._snp = subset_snp

            with dask.config.set(**{"array.slicing.split_large_chunks": False}):
                self._xr = dset_ref.xr.isel(snp=snp_idx, indiv=indiv_idx)

        else:
            # initialize from actual data set

            # assign `geno` and `lanc`
            assert geno is not None, "`geno` must not be None"
            data_vars: Dict[Hashable, Any] = {}
            data_vars["geno"] = (("snp", "indiv", "ploidy"), geno)

            n_snp, n_indiv = geno.shape[0:2]
            if lanc is not None:
                assert geno.shape == lanc.shape
                assert isinstance(lanc, da.Array), "`lanc` must be a dask array"
                data_vars["lanc"] = (("snp", "indiv", "ploidy"), lanc)

            # assign `indiv` and `snp`
            if snp is None:
                self._snp = pd.DataFrame(index=pd.RangeIndex(stop=n_snp))
            else:
                self._snp = snp

            if indiv is None:
                self._indiv = pd.DataFrame(index=pd.RangeIndex(stop=n_indiv))
            else:
                self._indiv = indiv

            self._xr = xr.Dataset(
                data_vars=data_vars,
                coords={"snp": self._snp.index, "indiv": self._indiv.index},
            )

            # assign attributes
            if ("lanc" in self._xr) and (n_anc is None):
                # infer number of ancestors
                n_anc = int(lanc[0:1000, :, :].max().compute() + 1)
                print(
                    "admix.Dataset: `n_anc` is not provided"
                    + f", infered n_anc from the first 1,000 SNPs is {n_anc}. "
                    + "If this is not correct, provide `n_anc` when constructing admix.Dataset"
                )
                self._xr.attrs["n_anc"] = n_anc
            else:
                self._xr.attrs["n_anc"] = n_anc

            self._xr.attrs["path"] = None
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
        descr = (
            f"admix.Dataset object with n_snp x n_indiv = {self.n_snp} x {self.n_indiv}"
        )
        if "lanc" in self._xr:
            descr += f", n_anc={self.n_anc}"
        else:
            descr += ", no local ancestry"

        if len(self.snp.columns) > 0:
            descr += "\n\tsnp: " + ", ".join([f"'{col}'" for col in self.snp.columns])
        if len(self.indiv.columns) > 0:
            descr += "\n\tindiv: " + ", ".join(
                [f"'{col}'" for col in self.indiv.columns]
            )

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
        return self._indiv

    @property
    def snp(self) -> pd.DataFrame:
        """One-dimensional annotation of observations (`pd.DataFrame`)."""
        return self._snp

    @property
    def geno(self) -> da.Array:
        """Genotype matrix"""
        return self._xr["geno"].data

    @property
    def lanc(self) -> da.Array:
        """Local ancestry matrix"""
        return self._xr["lanc"].data

    @property
    def xr(self) -> xr.Dataset:
        """Return the xr.Dataset used internally"""
        return self._xr

    def allele_per_anc(self) -> da.Array:
        """Return the allele-per-ancestry raw count matrix"""
        return admix.data.allele_per_anc(
            geno=self.geno, lanc=self.lanc, n_anc=self.n_anc, center=False
        )

    def af_per_anc(self, force=False) -> da.Array:
        """Return the allele-per-ancestry matrix"""
        if ("af_per_anc" not in self._xr) or force:
            self._xr["af_per_anc"] = ("snp", "anc"), admix.data.af_per_anc(
                geno=self.geno, lanc=self.lanc, n_anc=self.n_anc
            )
        return self._xr["af_per_anc"].data

    @property
    def uns(self) -> MutableMapping:
        """Unstructured annotation (ordered dictionary)."""
        uns = self._xr.attrs.get("uns", None)
        return uns

    def persist(self):
        """persist the lazy data to memory"""
        for name in ["geno", "lanc"]:
            if name in self._xr:
                self._xr[name] = (
                    self._xr[name].dims,
                    da.from_array(self._xr[name].data.compute(), chunks=-1),
                )

    def __getitem__(self, index) -> "Dataset":
        """Returns a sliced view of the object."""
        snp_idx, indiv_idx = normalize_indices(index, self.snp.index, self.indiv.index)
        return Dataset(dset_ref=self, snp_idx=snp_idx, indiv_idx=indiv_idx)

    def write(self, path):
        """Write admix.Dataset to disk

        Parameters
        ----------
        path : str
            path to the destiny place
        """
        # TODO: when writing to the


def subset_dataset(dset: Dataset, snp: List[str] = None, indiv: List[str] = None):
    """
    Read a dataset from a directory.
    """
    import dapgen

    if snp is not None:
        snp_mask = dset.snp.index.isin(snp)
    else:
        snp_mask = np.ones(len(dset.snp), dtype=bool)

    if indiv is not None:
        indiv_mask = dset.indiv.index.isin(indiv)
    else:
        indiv_mask = np.ones(len(dset.indiv), dtype=bool)

    return Dataset(
        geno=dset.geno[snp_mask, :, :][:, indiv_mask, :],
        lanc=dset.lanc[snp_mask, :, :][:, indiv_mask, :],
        snp=dset.snp.loc[snp_mask],
        indiv=dset.indiv.loc[indiv_mask],
        n_anc=dset.n_anc,
    )


def is_aligned(dset_list: List[Dataset], dim="snp"):
    """Check whether the datasets align with each other.

    Parameters
    ----------
    dset_list : List[Dataset]
        List of datasets to check
    dim: str
        Dimension to check. Either "snp" or "indiv"
    """
    if len(dset_list) == 0:
        return

    assert dim in ["snp", "indiv"], "dim must be either 'snp' or 'indiv'"

    if dim == "snp":
        df_snp_list = [dset.snp[["CHROM", "POS", "REF", "ALT"]] for dset in dset_list]
        return np.all([df_snp.equals(df_snp_list[0]) for df_snp in df_snp_list[1:]])
    elif dim == "indiv":
        df_indiv_list = [dset.indiv for dset in dset_list]
        return np.all(
            [df_indiv.equals(df_indiv_list[0]) for df_indiv in df_indiv_list[1:]]
        )
    else:
        raise ValueError("dim must be either 'snp' or 'indiv'")