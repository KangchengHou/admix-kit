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
    Any,
    Dict,
    Union,
    MutableMapping,
)
from ._index import normalize_indices
import warnings


class Dataset(object):
    """Data structure to contain genotype and local ancestry."""

    def __init__(
        self,
        geno: Optional[da.Array] = None,
        lanc: Optional[da.Array] = None,
        snp: Optional[pd.DataFrame] = None,
        indiv: Optional[pd.DataFrame] = None,
        n_anc: int = None,
        dset_ref=None,
        snp_idx: Union[slice, int, np.ndarray] = None,
        indiv_idx: Union[slice, int, np.ndarray] = None,
        enforce_order: bool = True,
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

            for name, idx in zip(["snp", "indiv"], [snp_idx, indiv_idx]):
                if isinstance(idx, slice):
                    assert (
                        (idx.start is None)
                        or (idx.stop is None)
                        or (idx.start < idx.stop)
                    ), f"Slice `{idx}` is not ordered."
                    if idx.step is not None:
                        assert idx.step > 0, f"Slice `{idx}` is not ordered."
                elif isinstance(idx, np.ndarray):
                    if enforce_order:
                        assert np.all(
                            idx == np.sort(idx)
                        ), f"idx=`{idx}` is not ordered according to dset.snp.index"
                else:
                    raise ValueError(
                        f"`{name}_idx` must be a slice or a numpy array of integers."
                    )
                if name == "snp":
                    df_subset = dset_ref.snp.iloc[idx, :].copy()
                    self._snp = df_subset
                elif name == "indiv":
                    df_subset = dset_ref.indiv.iloc[idx, :].copy()
                    self._indiv = df_subset
                else:
                    raise ValueError(f"Unknown name {name}")

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

        assert "lanc" in self._xr, "Local ancestry information is not available."
        return admix.data.allele_per_anc(
            geno=self.geno, lanc=self.lanc, n_anc=self.n_anc, center=False
        )

    def af_per_anc(self, force=False) -> da.Array:
        """
        Return the allele frequency per ancestry (n_snp, n_anc)

        nhaplo_per_anc will also be computed / updated.
        Parameters
        ----------
        force : bool
            If True, force re-computation of the matrix.
        """
        if ("af_per_anc" not in self._xr) or force:
            res = admix.data.af_per_anc(
                geno=self.geno,
                lanc=self.lanc,
                n_anc=self.n_anc,
                return_nhaplo=True,
            )
            self._xr["af_per_anc"] = ("snp", "anc"), res[0]
            self._xr["nhaplo_per_anc"] = ("snp", "anc"), res[1]

        return self._xr["af_per_anc"].data

    def nhaplo_per_anc(self, force=False) -> da.Array:
        """
        Return the number of haplotype per ancestry (n_snp, n_anc)

        af_per_anc will also be computed / updated.

        Parameters
        ----------
        force : bool
            If True, force re-computation of the matrix.
        """
        if ("nhaplo_per_anc" not in self._xr) or force:
            res = admix.data.af_per_anc(
                geno=self.geno,
                lanc=self.lanc,
                n_anc=self.n_anc,
                return_lanc_count=True,
            )
            self._xr["af_per_anc"] = ("snp", "anc"), res[0]
            self._xr["nhaplo_per_anc"] = ("snp", "anc"), res[1]

        return self._xr["nhaplo_per_anc"].data

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

    def append_indiv_info(
        self, df_info: pd.DataFrame, force_update: bool = False
    ) -> None:
        """
        append indiv info to the dataset, individual is matched using the self.indiv.index
        and df_info.index. Missing individuals in df_info will be filled with NaN.

        Parameters
        ----------
        df_info : pd.DataFrame
            DataFrame with the indiv info
        force_update : bool
            If True, update the indiv information even if it already exists.
        """
        if len(set(df_info.index) - set(self.indiv.index)) > 0:
            admix.logger.warn(
                "admix.dataset.append_indiv_info: "
                f"{len(set(df_info.index) - set(self.indiv.index))}/{len(set(df_info.index))}"
                " individuals in the new dataframe not in the dataset;"
                " These individuals will be ignored."
            )
        if len(set(self.indiv.index) - set(df_info.index)) > 0:
            admix.logger.warn(
                "admix.dataset.append_indiv_info: "
                f"{len(set(self.indiv.index) - set(df_info.index))}/{len(set(self.indiv.index))}"
                " individuals in the dataset are missing in the provided data frame."
                " These individuals will be filled with NaN."
            )

        df_info = df_info.reindex(self.indiv.index)

        # for every column in df_info, if it is not in the dataset, add it to the dataset
        # else, check the consistency of between df_info and dataset, if not consistent, raise error
        for col in df_info.columns:
            if col in self.indiv.columns:
                is_allclose = np.allclose(self.indiv[col], df_info[col], equal_nan=True)
                if not is_allclose:
                    if force_update:
                        admix.logger.info(
                            f"admix.dataset.append_indiv_info: "
                            f"{col} is updated from {self.indiv[col].values[0:5]} ..."
                            f"to {df_info[col].values[0:5]} ..."
                        )
                        self._indiv[col] = df_info[col]
                    else:
                        raise ValueError(
                            "admix.dataset.append_indiv_info: "
                            f"The column '{col}' in the provided data frame is not consistent "
                            "with the dataset."
                        )
            else:
                self._indiv[col] = df_info[col]

    def append_snp_info(self, df_info: pd.DataFrame) -> None:
        """
        append snp info to the dataset, snp is matched using the self.snp.index
        and df_info.index.

        Parameters
        ----------
        df_info : pd.DataFrame
            DataFrame with the snp info
        """
        if len(set(df_info.index) - set(self.snp.index)) > 0:
            # raise warning
            warnings.warn("Some SNPs in the `df_info` are not in the dataset.")

        df_info = df_info.reindex(self.snp.index)

        # for every column in df_info, if it is not in the dataset, add it to the dataset
        # else, check the consistency of between df_info and dataset, if not consistent, raise error
        for col in df_info.columns:
            if col in self.snp.columns:
                assert np.allclose(
                    self.snp[col], df_info[col]
                ), f"The column {col} in the `df_info` is not consistent with the dataset."
            else:
                self._snp[col] = df_info[col]

    def __getitem__(self, index) -> "Dataset":
        """Returns a sliced view of the object."""
        snp_idx, indiv_idx = normalize_indices(index, self.snp.index, self.indiv.index)
        return Dataset(dset_ref=self, snp_idx=snp_idx, indiv_idx=indiv_idx)


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
