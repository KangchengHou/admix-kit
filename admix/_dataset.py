import os
import pandas as pd
import numpy as np
import xarray as xr
import dask.array as da
from xarray.core.dataset import DataVariables
import dask
from typing import (
    Hashable,
    List,
    Optional,
    Any,
    Dict,
    Union,
    MutableMapping,
    Sequence,
    Tuple,
)

import warnings
from ._logging import logger

REQUIRED_SNP_COLUMNS = ["CHROM", "POS", "REF", "ALT"]


class Dataset(object):
    """
    Class to handle the dataset composed on genotype and local ancestry.

    An admix.Dataset `dset` Support the following operations:

    - dset.indiv[new_col] = new_values
    - dset.snp[new_col] = new_values

    Design principles:
    Use admix.Dataset to take charge of `geno` and `lanc` and arrays with >= 2 dimensions
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
        import admix

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
            import admix

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
            import admix

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
            logger.warn(
                "admix.Dataset.append_indiv_info: "
                f"{len(set(df_info.index) - set(self.indiv.index))}/{len(set(df_info.index))}"
                " individuals in the new dataframe not in the dataset;"
                " These individuals will be ignored."
            )
        if len(set(self.indiv.index) - set(df_info.index)) > 0:
            logger.warn(
                "admix.Dataset.append_indiv_info: "
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
                        logger.info(
                            f"admix.Dataset.append_indiv_info: "
                            f"{col} is updated from {self.indiv[col].values[0:5]} ..."
                            f"to {df_info[col].values[0:5]} ..."
                        )
                        self._indiv[col] = df_info[col]
                    else:
                        raise ValueError(
                            "admix.Dataset.append_indiv_info: "
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


##################################################################################
################################ index function ##################################
##################################################################################


def normalize_indices(
    index, snp_names: pd.Index, indiv_names: pd.Index
) -> Tuple[Union[slice, int, np.ndarray], Union[slice, int, np.ndarray]]:
    """Normalize the indices to return the snp slices, and individual slices

    Parameters
    ----------
    index : [type]
        [description]
    snp_names : pd.index
        [description]
    indiv_names : pd.index
        [description]

    Returns
    -------
    [type]
        [description]

    Raises
    ------
    ValueError
        [description]
    """
    # deal with tuples of length 1
    if isinstance(index, tuple) and len(index) == 1:
        index = index[0]

    if isinstance(index, tuple):
        if len(index) > 2:
            raise ValueError(
                "data can only be sliced in SNPs (first dim) and individuals (second dim)"
            )

    snp_ax, indiv_ax = unpack_index(index)
    snp_ax = _normalize_index(snp_ax, snp_names)
    indiv_ax = _normalize_index(indiv_ax, indiv_names)
    return snp_ax, indiv_ax


# convert the indexer (integer, slice, string, array) to the actual positions
# reference: https://github.com/theislab/anndata/blob/566f8fe56f0dce52b7b3d0c96b51d22ea7498156/anndata/_core/index.py#L16
def _normalize_index(
    indexer: Union[
        slice,
        int,
        str,
        np.ndarray,
    ],
    index: pd.Index,
) -> Union[slice, int, np.ndarray]:  # ndarray of int
    """Convert the indexed (integer, slice, string, array) to the actual positions

    Parameters
    ----------
    indexer : Union[ slice, int, str, np.ndarray, ]
        [description]
    index : pd.Index
        [description]

    Returns
    -------
    Union[slice, int, np.ndarray]
        [description]

    """
    if not isinstance(index, pd.RangeIndex):
        assert (
            index.dtype != float and index.dtype != int
        ), "Don't call _normalize_index with non-categorical/string names"

    # the following is insanely slow for sequences,
    # we replaced it using pandas below
    def name_idx(i):
        if isinstance(i, str):
            i = index.get_loc(i)
        return i

    if isinstance(indexer, slice):
        start = name_idx(indexer.start)
        stop = name_idx(indexer.stop)
        # string slices can only be inclusive, so +1 in that case
        if isinstance(indexer.stop, str):
            stop = None if stop is None else stop + 1
        step = indexer.step
        return slice(start, stop, step)
    elif isinstance(indexer, (np.integer, int)):
        return indexer
    elif isinstance(indexer, str):
        return index.get_loc(indexer)  # int
    elif isinstance(indexer, (Sequence, np.ndarray, pd.Index, np.matrix)):
        if hasattr(indexer, "shape") and (
            (indexer.shape == (index.shape[0], 1))
            or (indexer.shape == (1, index.shape[0]))
        ):
            indexer = np.ravel(indexer)
        if not isinstance(indexer, (np.ndarray, pd.Index)):
            indexer = np.array(indexer)
        if issubclass(indexer.dtype.type, (np.integer, np.floating)):
            return indexer  # Might not work for range indexes
        elif issubclass(indexer.dtype.type, np.bool_):
            if indexer.shape != index.shape:
                raise IndexError(
                    f"Boolean index does not match Dataset's shape along this "
                    f"dimension. Boolean index has shape {indexer.shape} while "
                    f"Dataset index has shape {index.shape}."
                )
            positions = np.where(indexer)[0]
            return positions  # np.ndarray[int]
        else:  # indexer should be string array
            positions = index.get_indexer(indexer)
            if np.any(positions < 0):
                not_found = indexer[positions < 0]
                raise KeyError(
                    f"Values {list(not_found)}, from {list(indexer)}, "
                    "are not valid obs/ var names or indices."
                )
            return positions  # np.ndarray[int]
    else:
        raise IndexError(f"Unknown indexer {indexer!r} of type {type(indexer)}")


def unpack_index(index):
    if not isinstance(index, tuple):
        return index, slice(None)
    elif len(index) == 2:
        return index
    elif len(index) == 1:
        return index[0], slice(None)
    else:
        raise IndexError("invalid number of indices")


def all_array_equal(array_list: List[np.ndarray]) -> bool:
    """check whether all arrays in the list are equal

    Parameters
    ----------
    List[np.ndarray]

    Returns
    -------
    bool
    """
    return all([np.allclose(array_list[0], a) for a in array_list[1:]])


def align_datasets(dsets: List[Dataset], dim: str) -> List[Dataset]:
    """takes 2 or more datasets, return the aligned dataset

    SNP alignment is based on index, CHROM, POS, REF, ALT
    Individual alignment is based on index

    TODO: refer to PLINK2 alignment page for preprocessing your data to align SNP attributes
    before hand.

    Parameters
    ----------
    dsets : List[Dataset]
        List of datasets
    dim : str
        which dimension to check

    Returns
    -------
    List[admix.Dataset]: list of aligned datasets
    """
    assert dim in ["snp", "indiv"]
    if dim == "snp":
        required_snp_columns = REQUIRED_SNP_COLUMNS
        # TODO: find aligned snp index among datasets


def check_align(dsets, dim: str) -> bool:
    """takes 2 or more datasets, and check whether the common attributes among datasets
    are properly aligned

    Parameters
    ----------
    dsets : List[admix.Dataset]
        List of datasets
    dim : str
        which dimension to check

    Returns
    -------
    bool: whether the two datasets align in the given dimension
    """
    aligned = False
    assert dim in ["snp", "indiv"], "dim must be either 'snp' or 'indiv'"
    if dim == "snp":
        required_snp_columns = REQUIRED_SNP_COLUMNS

        # find common columns among datasets
        common_cols = set.intersection(*[set(dset.snp.columns) for dset in dsets])

        # common_cols must be a subset of required_snp_columns
        assert set(required_snp_columns).issubset(
            common_cols
        ), "common_cols must be a subset of required_snp_columns"

        inconsistent_cols = [
            col
            for col in common_cols
            if not all_array_equal([dset.snp[col].values for dset in dsets])
        ]
        if not all_array_equal([dset.snp.index.values for dset in dsets]):
            inconsistent_cols.append("index")
        if len(inconsistent_cols) > 0:
            print(f"inconsistent columns: {','.join(inconsistent_cols)} in snp")
            aligned = False
        else:
            aligned = True

    elif dim == "indiv":
        # find common columns among datasets
        common_cols = set.intersection(*[set(dset.indiv.columns) for dset in dsets])
        # check whether all datasets have the same columns
        inconsistent_cols = [
            col
            for col in common_cols
            if not all_array_equal([dset.indiv[col].values for dset in dsets])
        ]
        if len(inconsistent_cols) > 0:
            print(f"inconsistent columns: {','.join(inconsistent_cols)} in indiv")
            aligned = False
        else:
            aligned = True
    return aligned


##################################################################################
################################## toy data ######################################
##################################################################################


def get_test_data_dir() -> str:
    """
    Get toy dataset directory

    Returns
    -------
    str
        Toy dataset directory
    """
    test_data_path = os.path.join(os.path.dirname(__file__), "../tests/test-data")
    return test_data_path


def load_toy_admix() -> Dataset:
    """
    Load toy admixed data set with African and European ancestries

    Returns
    -------
    Dataset
    """
    from .io import read_dataset

    dset = read_dataset(os.path.join(get_test_data_dir(), "toy-admix"), n_anc=2)
    return dset


def load_toy() -> List[Dataset]:
    """Load toy dataset

    Load simulated
    (1) 50 admixed individuals
    (2) 50 EUR individuals
    (3) 50 AFR individuals

    5000 SNPs

    Returns
    -------
    List[admix.Dataset]
        [dset_admix, dset_eur, dset_afr]
    """

    # TODO: change the data format, use .pgen and .lanc
    import xarray as xr

    module_path = os.path.dirname(__file__)
    test_data_path = os.path.join(module_path, "../tests/test-data")
    dset_eur = xr.open_zarr(os.path.join(test_data_path, "eur.zip"))
    dset_afr = xr.open_zarr(os.path.join(test_data_path, "afr.zip"))
    dset_admix = xr.open_zarr(os.path.join(test_data_path, "admix.zip"))

    dset_list = [
        Dataset(
            geno=np.swapaxes(dset_admix.geno.data, 0, 1),
            lanc=np.swapaxes(dset_admix.lanc.data, 0, 1),
            n_anc=2,
            indiv=dset_admix.indiv.to_dataframe().drop(columns=["indiv"]),
            snp=dset_admix.snp.to_dataframe().drop(columns=["snp"]),
        )
    ]
    for dset in [dset_eur, dset_afr]:
        dset_list.append(
            Dataset(
                geno=np.swapaxes(dset.geno.data, 0, 1),
                n_anc=1,
                indiv=dset.indiv.to_dataframe().drop(columns=["indiv"]),
                snp=dset.snp.to_dataframe().drop(columns=["snp"]),
            ),
        )

    return dset_list