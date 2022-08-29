import dask.array as da
import numpy as np
import admix
from typing import List, Tuple, Union, Dict, Sequence
from dask.array import concatenate, from_delayed
from dask.delayed import delayed
from bisect import bisect_left, bisect_right
import dask

is_sorted = lambda a: np.all(a[:-1] <= a[1:])


class Lanc(object):
    """
    Class for local ancestry matrix (n_snp, n_indiv, 2 x ploidy)
    The internal representation is sparse using `breaks` and `values`
    """

    def __init__(
        self,
        path: str = None,
        breaks: List[List[int]] = None,
        values: List[List[str]] = None,
        array: Union[da.Array, np.ndarray] = None,
    ):
        # only one of `path`, `breaks` and `values` can be specified
        assert (path is not None) + ((breaks is not None) & (values is not None)) + (
            array is not None
        ) == 1, "Only one of `path`, `breaks` & `values`, and `array` can be specified"
        if path is not None:
            breaks, values = read_lanc(path)
        elif array is not None:
            breaks, values = array_to_lanc(array)

        assert breaks is not None and values is not None
        n_snp, n_indiv = check_lanc_format(breaks, values)

        self._n_snp = n_snp
        self._n_indiv = n_indiv

        self._breaks = breaks
        self._values = values

    def __repr__(self) -> str:
        descr = (
            f"admix.Lanc object with n_snp x n_indiv = {self.n_snp} x {self.n_indiv}"
        )
        descr += f", use `lanc.dask()` to convert to dask array."

        return descr

    @property
    def n_indiv(self) -> int:
        """Number of individuals."""
        return self._n_indiv

    @property
    def n_snp(self) -> int:
        """Number of SNPs."""
        return self._n_snp

    def lanc_count(self, n_anc: int = None) -> np.ndarray:
        """
        Count the number of local ancestries for each individual.

        Parameters
        ----------
        n_anc : int
            Number of local ancestries in the local ancestries matrix. If None,
            the number of local ancestries is inferred from the data.

        Returns
        -------
            Number of local ancestries for each individual.
        """

        breaks, values = self._breaks, self._values
        n_indiv = self.n_indiv

        # the maximal number of ancestries is assumed to be 10
        MAX_ANC = 10
        lanc_count = np.zeros((n_indiv, MAX_ANC), dtype=int)
        if n_anc is not None:
            lanc_count = lanc_count[:, :n_anc]

        for indiv_i in range(n_indiv):
            start = 0
            for stop, val in zip(breaks[indiv_i], values[indiv_i]):
                a1, a2 = int(val[0]), int(val[1])
                lanc_count[indiv_i, a1] += stop - start
                lanc_count[indiv_i, a2] += stop - start
                start = stop

        # remove any column with zero entries
        if n_anc is None:
            n_anc = np.argmax(np.sum(lanc_count, axis=0) == 0)
        lanc_count = lanc_count[:, :n_anc]
        return lanc_count

    def impute(self, chrom_pos: np.ndarray, dst_chrom_pos: np.ndarray) -> "Lanc":
        """
        Impute missing values in `pos` using `target_pos`.

        Parameters
        ----------
        chrom_pos : np.ndarray
            Chromosomes and positions of the source data.
        dst_chrom_pos
            Chromosomes and positions of the destination data.

        Returns
        -------
        imputed : Lanc
        """
        assert (chrom_pos.shape[1] == 2) and (dst_chrom_pos.shape[1] == 2)
        assert chrom_pos.shape[0] == self.n_snp
        assert is_sorted(chrom_pos[:, 0]) & is_sorted(dst_chrom_pos[:, 0])
        assert set(chrom_pos[:, 0]) == set(
            dst_chrom_pos[:, 0]
        ), "Chromosome set must be the same"
        chrom_list = sorted(set(chrom_pos[:, 0]))

        dst_lanc_list = []
        for chrom in chrom_list:
            src_chrom_idx = np.where(chrom_pos[:, 0] == chrom)[0]
            dst_dst_idx = np.where(dst_chrom_pos[:, 0] == chrom)[0]
            src_chrom_start = src_chrom_idx[0]
            src_chrom_stop = src_chrom_idx[-1] + 1

            chrom_src_breaks, chrom_src_values = lanc_subset_snp_range(
                self._breaks, self._values, src_chrom_start, src_chrom_stop
            )

            chrom_dst_breaks, chrom_dst_values = lanc_impute_single_chrom(
                chrom_src_breaks,
                chrom_src_values,
                chrom_pos[src_chrom_idx, 1],
                dst_chrom_pos[dst_dst_idx, 1],
            )

            dst_lanc_list.append(Lanc(breaks=chrom_dst_breaks, values=chrom_dst_values))
        return concat_lancs(dst_lanc_list)

    def dask(self, snp_chunk: int = 1024):
        return lanc_to_dask(self._breaks, self._values, snp_chunk)

    def numpy(self):
        return lanc_to_numpy(self._breaks, self._values)

    def __getitem__(self, index) -> "Lanc":
        """
        Returns a new instance of `Lanc` with the specified index.
        First index is the SNP, second index is the individual.
        The SNP index can only be a slice with `step` = 1.
        The individual index can be a list of integers or slice, etc.
        """
        if isinstance(index, tuple) and len(index) == 1:
            index = index[0]

        if isinstance(index, tuple):
            if len(index) > 2:
                raise ValueError(
                    "Lanc can only be sliced in SNPs (first dim) and individuals (second dim)"
                )

        from ..dataset._index import unpack_index

        snp_idx, indiv_idx = unpack_index(index)
        # if indiv_idx is slice(None), use all individuals
        if indiv_idx is slice(None):
            indiv_idx = slice(0, self.n_indiv)
        assert isinstance(snp_idx, slice), "SNP index must be a slice"
        assert (snp_idx.step is None) or (
            snp_idx.step is None
        ), "SNP index must with `step` = 1"
        snp_start = snp_idx.start if snp_idx.start is not None else 0
        snp_stop = snp_idx.stop if snp_idx.stop is not None else self.n_snp
        assert snp_start >= 0, "SNP index must start at 0 or higher"
        assert snp_stop <= self.n_snp, "SNP index must stop at n_snp or lower"
        assert snp_start < snp_stop, "SNP index must be a slice with `start < stop`"

        breaks = self._breaks[indiv_idx]
        values = self._values[indiv_idx]
        breaks, values = lanc_subset_snp_range(breaks, values, snp_start, snp_stop)
        return Lanc(breaks=breaks, values=values)

    def write(
        self,
        path: str,
    ):
        breaks, values = self._breaks, self._values
        n_snp, n_indiv = check_lanc_format(breaks, values)

        # write to file
        lines = []

        # header
        lines.append(f"{n_snp} {n_indiv}")

        for indiv_break, indiv_value in zip(breaks, values):
            lines.append(
                " ".join([str(b) + ":" + v for (b, v) in zip(indiv_break, indiv_value)])
            )

        with open(path, "w") as f:
            f.writelines("\n".join(lines))


def assign_lanc(dset: admix.Dataset, lanc_file: str, format: str = "rfmix"):
    """
    Assign local ancestry to a dataset. Currently we assume that the rfmix file contains
    2-way admixture information.

    Parameters
    ----------
    dset: admix.Dataset
        Dataset to assign local ancestry to.
    lanc_file: str
        Path to local ancestry data.
    format: str
        Format of local ancestry data.
        Currently only "rfmix" is supported.

    Returns
    -------
    dset: admix.Dataset
        Dataset with local ancestry assigned.
    TODO:
    - Add support for other formats.
    """
    import pandas as pd
    import numpy as np

    assert format in ["rfmix"], "Only rfmix format is supported."
    # assign local ancestry
    rfmix = pd.read_csv(lanc_file, sep="\t", skiprows=1)

    lanc_full = da.full(
        shape=(dset.dims["indiv"], dset.dims["snp"], dset.dims["ploidy"]),
        fill_value=-1,
        dtype=np.int8,
    )
    lanc0 = rfmix.loc[:, rfmix.columns.str.endswith(".0")].rename(
        columns=lambda x: x[:-2]
    )
    lanc1 = rfmix.loc[:, rfmix.columns.str.endswith(".1")].rename(
        columns=lambda x: x[:-2]
    )

    assert np.all(dset.indiv == lanc0.columns)
    assert np.all(dset.indiv == lanc1.columns)

    for i_row, row in rfmix.iterrows():
        mask_row = (
            (row.spos <= dset.snp["POS"]) & (dset.snp["POS"] <= row.epos)
        ).values
        lanc_full[:, mask_row, 0] = lanc0.loc[i_row, :].values[:, np.newaxis]
        lanc_full[:, mask_row, 1] = lanc1.loc[i_row, :].values[:, np.newaxis]

    dset_names = tuple(d for d in dset.dims)
    if dset_names == ("indiv", "snp", "ploidy"):
        # do nothing
        pass
    elif dset_names == ("snp", "indiv", "ploidy"):
        lanc_full = lanc_full.swapaxes(0, 1)
    else:
        raise ValueError(
            f"Unexpected dimensions {dset_names}. "
            "Expected (indiv, snp, ploidy) or (snp, indiv, ploidy)"
        )

    lanc_full = lanc_full.rechunk(dset.geno.chunks)

    dset = dset.assign({"lanc": (dset_names, lanc_full)})
    dset = dset.assign_attrs({"n_anc": 2})
    return dset


def find_closest_index(a, x, tie="left"):
    """
    Locate the index in a of the closest value to x.
    When two values are equally close, return the smallest index.

    Parameters
    ----------
    a: array-like
        Values to search.
    x: float
        Value to search for.
    tie: str
        How to handle ties. left -> return the smallest index.
        right -> return the largest index. default: left
    """
    assert tie in ["left", "right"], "tie must be 'left' or 'right'"
    if tie == "left":
        op = np.greater_equal
    else:
        op = np.greater

    idx = bisect_left(a, x)

    if idx == len(a):
        # x is larger than a[-1]
        idx = len(a) - 1
    elif idx > 0 and op(a[idx] - x, x - a[idx - 1]):
        # x is closer to a[idx-1] than a[idx]
        idx -= 1
    return idx


def read_bp_lanc(path: str):
    """
    read .bp file (breakpoints file) and convert to lanc object.
    The format is specified at
    https://github.com/williamslab/admix-simu#output-breakpoints-file-out_prefixbp

    Parameters
    ----------
    path : str
        path to .bp file

    Returns
    -------
    admix.data.Lanc
        the converted local ancestry

    """
    assert path.endswith(".bp")
    with open(path, "r") as f:
        bp_data = [line.strip().split() for line in f.readlines()][1:]
    hap_breaks = [[int(l.split(":")[1]) + 1 for l in line] for line in bp_data]
    hap_values = [[l.split(":")[0] for l in line] for line in bp_data]

    dip_breaks, dip_values = clean_lanc(
        *admix.data.haplo2diplo(breaks=hap_breaks, values=hap_values),
        remove_repeated_val=True,
    )
    lanc = admix.data.Lanc(breaks=dip_breaks, values=dip_values)
    return lanc


def concat_lancs(lanc_list: List[Lanc], dim="snp") -> Lanc:
    """Concatenate local ancestry files
    For example (1:01, 15: 00) and (3:01, 12:01) will be merged to (1:01, 15:00, 18:01, 27:01)
    Parameters
    ----------
    lanc_list: List[Lanc]
        List of local ancestry files.

    Returns
    -------
    """
    assert dim == "snp", "Only concatenating along snp is supported."

    snp_start = 0
    n_indiv = lanc_list[0].n_indiv
    breaks: List[List[int]] = [[] for _ in range(n_indiv)]
    values: List[List[str]] = [[] for _ in range(n_indiv)]
    for lanc in lanc_list:
        assert lanc.n_indiv == n_indiv, "Number of individuals must be the same."
        for indiv_i in range(n_indiv):
            breaks[indiv_i].extend([b + snp_start for b in lanc._breaks[indiv_i]])
            values[indiv_i].extend(lanc._values[indiv_i])
        snp_start += lanc.n_snp
    return Lanc(breaks=breaks, values=values)


def read_lanc(path: str) -> Tuple[List, List]:
    """Read local ancestry with .lanc format

    Parameters
    ----------
    path : .lanc file
        Local ancestry file

    Returns
    -------
    da.Array
        (n_snp, n_indiv, 2)
    """

    # TODO: first check input
    # the end must equal to n_snp
    with open(path) as f:
        lines = f.readlines()
    n_snp0, n_indiv0 = [int(i) for i in lines[0].strip().split()]
    data_list = [line.strip().split() for line in lines[1:]]
    assert len(data_list) == n_indiv0
    breaks = [[int(l.split(":")[0]) for l in line] for line in data_list]
    values = [[l.split(":")[1] for l in line] for line in data_list]

    n_snp, n_indiv = check_lanc_format(breaks, values)
    assert n_snp0 == n_snp, "n_snp in .lanc file does not match"
    assert n_indiv0 == n_indiv, "n_indiv in .lanc file does not match"
    return breaks, values


def check_lanc_format(breaks: List[List[int]], values: List[List[str]]):
    """Check the format of .lanc file

    Parameters
    ----------
    breaks : List[List[int]]
        List of break points
    values : List[List[str]]
        List of value

    Returns
    -------
    Tuple[int, int]
        (n_snp, n_indiv)
    """
    assert len(breaks) == len(
        values
    ), "`breaks` and `values` must have the same length (same as n_indiv)"

    assert np.all([len(b) == len(v) for b, v in zip(breaks, values)])
    n_snp = breaks[0][-1]
    n_indiv = len(breaks)
    assert np.all(
        [n_snp == b[-1] for b in breaks]
    ), "The last element of break points for each individual in `break_list` must be equal"
    return n_snp, n_indiv


def clean_lanc(
    breaks: List[List[int]], values: List[List[str]], remove_repeated_val: bool = False
):
    """Clean up local ancestry file

    Parameters
    ----------
    breaks : List[List[int]]
        break points
    values : List[List[str]]
        values
    remove_same_val : bool
        Remove segments with same values (default: False)
        For example, 50:01 100:01 300:00 -> 100:01 300:00
        For example, 50:01 100:01 100:10 300:00 300:01 -> 100:01 300:00

    """
    new_breaks = []
    new_values = []

    if not remove_repeated_val:
        for br, vl in zip(breaks, values):
            # remove duplicated break positions, only preserve the first one
            d = dict()
            for b, v in zip(br, vl):
                if b not in d:
                    d[b] = v

            d.pop(0, None)
            new_breaks.append(list(d.keys()))
            new_values.append(list(d.values()))
    else:
        for br, vl in zip(breaks, values):
            # remove duplicated break positions, only preserve the first one
            last_break = None
            last_value = None
            d = dict()
            for b, v in zip(br, vl):
                if (last_value is not None) and (v != last_value):
                    # this value is different from the last one
                    # record last one as break points
                    if last_break not in d:
                        d[last_break] = last_value
                # either: last_value is None
                # or: this value is the same as the last one, extend
                last_break, last_value = b, v

            # record finel one
            assert last_break is not None
            if last_break not in d:
                d[last_break] = last_value

            d.pop(0, None)
            new_breaks.append(list(d.keys()))
            new_values.append(list(d.values()))
    return new_breaks, new_values


def haplo2diplo(breaks: List[List], values: List[List]):
    """convert haplotype to diplotype, combine every 2 haplotypes into 1 diplotype

    Parameters
    ----------
    breaks : List[List[int]]
        break points
    values : List[List[str]]
        values

    Returns
    -------
    List[List[int]]
        break points
    List[List[str]]
        values
    """
    assert len(breaks) == len(values)
    n_haplo = len(breaks)
    assert n_haplo % 2 == 0, "Number of haplotypes must be even"
    n_diplo = n_haplo // 2

    new_breaks = []
    new_values = []
    for i in range(n_diplo):
        br1, br2 = breaks[i * 2 : (i + 1) * 2]
        vl1, vl2 = values[i * 2 : (i + 1) * 2]
        unique_br = np.union1d(br1, br2).tolist()
        new_vl = [
            str(v1) + str(v2)
            for v1, v2 in zip(
                [vl1[bisect_left(br1, b)] for b in unique_br],
                [vl2[bisect_left(br2, b)] for b in unique_br],
            )
        ]
        new_breaks.append(unique_br)
        new_values.append(new_vl)
    return new_breaks, new_values


def lanc_subset_snp_range(
    breaks: List[List[int]], values: List[List[str]], start: int, stop: int
):
    """
    Subset the .lanc file

    Parameters
    ----------
    start : int
        start of SNP
    stop : int
        stop of SNP
    break_list: List[List[int]]
        list of break points
    value_list: List[List[str]]
    """
    # TODO: double check this function whether usage of bisect_left is correct
    # For each individual, find index of break points that's within [start, stop]
    start_idx = [bisect_left(indiv_break, start) for indiv_break in breaks]
    stop_idx = [bisect_left(indiv_break, stop) for indiv_break in breaks]
    new_breaks = [br[s:e] + [stop] for s, e, br in zip(start_idx, stop_idx, breaks)]
    # offset with start
    new_breaks = [[b - start for b in br] for br in new_breaks]
    # find corresponding value
    new_values = [val[s:e] + [val[e]] for s, e, val in zip(start_idx, stop_idx, values)]

    # clean up
    new_breaks, new_values = clean_lanc(new_breaks, new_values)
    return new_breaks, new_values


def lanc_to_dask(
    breaks: List[List[int]], values: List[List[str]], snp_chunk: int = 1024
):
    """
    Given `break_list` list of break points with `n_indiv` length
    And the corresponding `value_list`, the correponding value

    Convert to dask matrix
    """
    n_snp, n_indiv = check_lanc_format(breaks, values)

    # all local ancestries
    lancs = []

    subset_dense = lambda start, stop: lanc_to_numpy(
        *lanc_subset_snp_range(breaks, values, start, stop)
    )
    snp_start = 0
    while snp_start < n_snp:
        snp_stop = min(snp_start + snp_chunk, n_snp)
        shape = (snp_stop - snp_start, n_indiv, 2)

        lancs.append(
            from_delayed(
                value=delayed(subset_dense)(snp_start, snp_stop),
                shape=shape,
                dtype=np.int8,
            )
        )
        snp_start = snp_stop
    return concatenate(lancs, 0, False)


def lanc_to_numpy(breaks, values):
    """
    Given `break_list` list of break points with `n_indiv` length
    And the corresponding `value_list`, the correponding value

    Convert to dense matrix
    """
    n_snp, n_indiv = check_lanc_format(breaks, values)
    mat = np.full((n_snp, n_indiv, 2), -1)
    for indiv_i in range(n_indiv):
        start = 0
        for stop, val in zip(breaks[indiv_i], values[indiv_i]):
            a1, a2 = int(val[0]), int(val[1])
            mat[start:stop, indiv_i, 0] = a1
            mat[start:stop, indiv_i, 1] = a2
            start = stop
    return mat


def array_to_lanc(array: Union[da.Array, np.ndarray]) -> Tuple[List, List]:
    """Convert dense array to local ancestry format

    Parameters
    ----------
    array : np.ndarray
        (n_snp, n_indiv, 2)

    Returns
    """
    n_snp, n_indiv, n_ploidy = array.shape
    assert n_ploidy == 2, "`lanc` must be (n_snp, n_indiv, 2)"

    # convert to dask.array if numpy array
    if isinstance(array, np.ndarray):
        lanc = da.from_array(array)
    else:
        lanc = array

    assert isinstance(lanc, da.Array), "`lanc` must be dask array"

    # switch points
    snp_pos, indiv_pos, _, = dask.compute(
        np.where(lanc[1:, :, :] != lanc[0:-1, :, :]),
        scheduler="single-threaded",
    )[0]
    # end points for all the individuals
    snp_pos = np.concatenate([snp_pos, [n_snp - 1] * n_indiv])
    indiv_pos = np.concatenate([indiv_pos, np.arange(n_indiv)])
    # (snp, indiv) -> snp * n_indiv + indiv
    values = lanc.reshape([-1, 2])[indiv_pos + snp_pos * n_indiv, :].compute()
    values = np.array([str(v[0]) + str(v[1]) for v in values])

    break_list = []
    value_list = []
    for indiv_i in range(n_indiv):
        indiv_mask = indiv_pos == indiv_i
        # +1 because .lanc denote the [start, stop) right-open interval
        indiv_snp_pos, unique_mask = np.unique(
            snp_pos[indiv_mask] + 1, return_index=True
        )
        indiv_values = values[indiv_mask][unique_mask]
        break_list.append(indiv_snp_pos.tolist())
        value_list.append(indiv_values.tolist())
    return break_list, value_list


def lanc_impute_single_chrom(
    src_breaks: List[List[int]],
    src_values: List[List[str]],
    src_pos: np.ndarray,
    dst_pos: np.ndarray,
):
    """local ancestry imputation for single chromosome

    Using the following steps:
        1. basic checks are performed for the two data sets.
        2. `dset_ref`'s individuals is matched with `dset`, `dset`'s individuals
            must be a subset of `dset_ref`'s individuals.
        3. Imputation is performed based on that, for each position in `dst`, find the
        nearest SNP in `src` measured by physical distance, and use the assignment in src.
        if there is a tie, use the one in `src` with the smallest index.

    Parameters
    ----------
    ref_pos : np.ndarray
        position in reference data sets
    ref_lanc : admix.data.Lanc
        reference local ancestry
    sample_pos : np.ndarray
        position in sample data sets to be imputed
    """
    src_n_snp, n_indiv = check_lanc_format(src_breaks, src_values)
    dst_n_snp = len(dst_pos)
    assert len(src_pos) == src_n_snp
    assert is_sorted(src_pos) & is_sorted(dst_pos), "src_pos and dst_pos must be sorted"
    assert (src_pos.ndim == 1) & (
        dst_pos.ndim == 1
    ), "src_pos and dst_pos must be 1-D arrays"
    assert np.issubdtype(src_pos.dtype, np.integer) & np.issubdtype(
        dst_pos.dtype, np.integer
    ), "pos and dst_pos must be integer type"

    # find physical position of break points in src data
    # `b - 1` because breaks corresponds to right-open interval

    dst_breaks = []
    dst_values = []
    for indiv_src_breaks, indiv_src_values in zip(src_breaks, src_values):
        # find a set of candidate index for dst break points
        # for each break points in src_pos, define `mid`` = (src_pos[b] + src_pos[b+1])/2
        # then find the closest point in dst that's smaller or equal to `mid`
        # then assign the local ancestry at b in src to that point
        # max(b - 1, 0) is to avoid accidental break = 0 (which should not appear)
        # we threshold by 0 anyway to avoid negative index
        indiv_dst_breaks = [
            bisect_right(
                dst_pos, (src_pos[max(b - 1, 0)] + src_pos[b]) / 2  # type: ignore
            )
            for b in indiv_src_breaks[:-1]
        ] + [dst_n_snp]
        dst_breaks.append(indiv_dst_breaks)
        dst_values.append(indiv_src_values)

    return clean_lanc(dst_breaks, dst_values)


###############################################################################
# LEGACY code for impute_lanc
###############################################################################

# version 2
# to be computational efficient, we start with src data
# for each src break point, find the closest dst break point
# dst_breaks = []
# dst_values = []
# for indiv_src_break_pos, indiv_src_values in zip(src_break_pos, src_values):
#     # find a set of candidate index for dst break points
#     indiv_dst_breaks = [
#         find_closest_index(dst_pos, p, tie="right") + 1
#         for p in indiv_src_break_pos[:-1]
#     ] + [dst_n_snp]
#     # assign src values to dst break points
#     indiv_dst_values = [
#         indiv_src_values[
#             find_closest_index(indiv_src_break_pos, dst_pos[i - 1], tie="left")
#         ]
#         for i in indiv_dst_breaks
#     ]

#     dst_breaks.append(indiv_dst_breaks)
#     dst_values.append(indiv_dst_values)

# version 1
# dict_tmp: Dict[int, Tuple] = {}
# for p, v in zip(indiv_src_break_pos[:-1], indiv_src_values[:-1]):
#     # tie is right such that for the dst, tie is always the one with the smallest index
#     idx = find_closest_index(dst_pos, p, tie="right")
#     dist = np.abs(dst_pos[idx] - p)
#     print(f"src_break_pos: {p}, src_value: {v}, dst_idx: {idx}, dist: {dist}")
#     # replace the dist, value if the new distance is smaller
#     if (idx not in dict_tmp) or (dist < dict_tmp[idx][0]):
#         dict_tmp[idx] = (dist, v)
# # the last src break point, always correspond to the last dst break point
# dst_breaks.append([k + 1 for k in dict_tmp.keys()] + [dst_n_snp])
# dst_values.append([v[1] for v in dict_tmp.values()] + [indiv_src_values[-1]])


# def impute_lanc(dset: admix.Dataset, dset_ref: admix.Dataset):
#     """
#     Impute local ancestry using a reference dataset. The two data sets are assumed to
#     have the same haplotype order, etc. Typically they are just a subset of each other.

#     Using the following steps:
#         1. basic checks are performed for the two data sets.
#         2. `dset_ref`'s individuals is matched with `dset`, `dset`'s individuals
#             must be a subset of `dset_ref`'s individuals.
#         3. Imputation are performed

#     Parameters
#     ----------
#     dset: a data set to be imputed with local ancestry
#     dset_ref: a data set with local ancestry for reference

#     Returns
#     -------
#     dset_imputed: a data set with imputed local ancestry
#     """
#     assert (
#         len(set(dset.snp.CHROM)) == 1
#     ), "Data set to be imputed can only have one chromosome"

#     assert (
#         len(set(dset_ref.snp.CHROM)) == 1
#     ), "Data set to be imputed can only have one chromosome"

#     # dset.indiv is a subset of dset_ref.indiv
#     assert np.all(
#         dset.indiv.index, dset_ref.indiv.index
#     ), "dset and dset_ref must have the same individuals"

#     # find relevant regions in reference dataset with local ancestry (hapmap3 SNPs here)
#     ref_start = np.argmin(np.abs(dset_ref.snp.POS.values - dset.snp.POS.values[0]))
#     ref_stop = np.argmin(np.abs(dset_ref.snp.POS.values - dset.snp.POS.values[-1]))

#     # subset reference dataset to relevant regions
#     dset_ref = dset_ref[ref_start : ref_stop + 1]

#     imputed_lanc = []
#     for ploidy_i in range(2):
#         # form a dataframe which contains the known local ancestry and locations to be imputed
#         df_snp = pd.concat([dset.snp[["POS"]], dset_ref.snp[["POS"]]])
#         df_snp = df_snp[~df_snp.index.duplicated()].sort_values("POS")

#         df_lanc = pd.DataFrame(
#             index=df_snp.index.values, columns=dset.indiv.index.values, dtype=float
#         )
#         # fill margin
#         df_lanc.iloc[0, :] = dset_ref.lanc[0, :, ploidy_i]
#         df_lanc.iloc[-1, :] = dset_ref.lanc[-1, :, ploidy_i]

#         # fill inside
#         df_lanc.loc[dset_ref.snp.values, :] = dset_ref["lanc"][:, :, ploidy_i].values

#         # interpolate
#         df_lanc = df_lanc.reset_index().interpolate(method="nearest").set_index("index")

#         imputed_lanc.append(df_lanc.loc[dset["snp"].values, :].values.astype(np.int8))

#     # imputed_lanc is in the order of ("snp", "indiv", "ploidy")
#     return da.from_array(np.dstack(imputed_lanc), chunks=-1)
