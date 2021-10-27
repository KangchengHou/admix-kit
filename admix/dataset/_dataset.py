import pandas as pd
import xarray as xr
import numpy as np
import dask.array as da
from xarray.core.dataset import DataVariables

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
        n_anc: int = None,
        uns: Optional[Mapping[str, Any]] = None,
    ):
        assert geno is not None
        data_vars: Dict[Hashable, Any] = {}
        data_vars["geno"] = (("snp", "indiv", "ploidy"), geno)

        n_snp, n_indiv = geno.shape[0:2]
        if lanc is not None:
            data_vars["lanc"] = (("snp", "indiv", "ploidy"), lanc)

        if indiv is None:
            self._indiv = pd.DataFrame(index=np.arange(n_indiv))
        else:
            self._indiv = indiv

        if snp is None:
            self._snp = pd.DataFrame(index=np.arange(n_snp))
        else:
            self._snp = snp

        self._xr = xr.Dataset(data_vars=data_vars)

        if ("lanc" in self._xr) and (n_anc is None):
            # infer number of ancestors
            n_anc = int(lanc.max().compute() + 1)
            self._xr.attrs["n_anc"] = n_anc
        else:
            self._xr.attrs["n_anc"] = n_anc

        self._path = None
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

        descr += "\n\tsnp: " + ", ".join([f"'{col}'" for col in self.snp.columns])
        descr += "\n\tindiv: " + ", ".join([f"'{col}'" for col in self.indiv.columns])

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
    def allele_per_anc(self) -> da.Array:
        """Return the allele-per-ancestry matrix"""
        return self._xr["allele_per_anc"].data

    @property
    def af_per_anc(self) -> da.Array:
        """Return the allele-per-ancestry matrix"""
        return self._xr["af_per_anc"].data

    @property
    def uns(self) -> MutableMapping:
        """Unstructured annotation (ordered dictionary)."""
        uns = self._xr.attrs.get("uns", None)
        return uns

    def load(self):
        """load the lazy data to memory"""
        for name in ["geno", "lanc"]:
            if name in self._xr:
                self._xr[name] = (
                    self._xr[name].dims,
                    da.from_array(self._xr[name].data.compute(), chunks=-1),
                )

    def compute_allele_per_anc(self, center: bool = False):
        """
        Compute number of minor alleles per ancestry

        Parameters
        ----------
        center : bool
            whether to center the number of minor alleles by the allele frequency
        """
        # TODO: if "af_per_anc" already in data set, this should be
        # fed into allele_per_anc.
        import admix

        self._xr["allele_per_anc"] = (
            "snp",
            "indiv",
            "anc",
        ), admix.data.allele_per_anc(
            geno=self.geno, lanc=self.lanc, center=center, n_anc=self.n_anc
        )

    def compute_af_per_anc(self):
        """
        Compute allele frequency per ancestry

        Parameters
        ----------
        center : bool
            whether to center the number of minor alleles by the allele frequency
        """
        import admix

        self._xr["af_per_anc"] = ("snp", "anc"), admix.data.af_per_anc(
            geno=self.geno, lanc=self.lanc, n_anc=self.n_anc
        )

    def __getitem__(self, index) -> "Dataset":
        """Returns a sliced view of the object."""
        assert False, "Not implemented"
        # snp_idx, indiv_idx = index
        # snp_idx, indiv_idx = self._normalize_indices(index)
        # return Dataset(geno, snp_idx=snp_idx, indiv_idx=indiv_idx)


def read_dataset(prefix, n_anc=2):
    """
    Read a dataset from a directory.
    """
    import dapgen
    import admix

    geno, pvar, psam = dapgen.read_pfile(prefix, phase=True)
    lanc = admix.io.read_lanc(prefix + ".lanc")
    # return geno, lanc
    return Dataset(geno=geno, lanc=lanc, snp=pvar, indiv=psam, n_anc=n_anc)


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


from os.path import dirname, join
