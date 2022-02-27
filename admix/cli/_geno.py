import os
from typing import List
import admix
import pandas as pd
import dapgen
import numpy as np
from ._utils import log_params


def _append_to_file(path: str, df: pd.DataFrame):
    """
    Append a new data frame to an existing .pvar file. The index of the new data frame
    is assumed to be exactly the same as the index of the existing data frame.

    Parameters
    ----------
    pvar_path : str
        Path to the .pvar file.
    df : pd.DataFrame
        Data frame to append.

    """
    # if path does not exist, directly write to it
    if not os.path.exists(path):
        df.to_csv(path, sep="\t", float_format="%.8g")
    else:
        df_snp_info = pd.read_csv(path, sep="\t", index_col=0)
        assert df_snp_info.index.equals(
            df.index
        ), f"SNPs in {path} and the provided data frame do not match"
        # no overlap of columns
        overlap_cols = set(df_snp_info.columns) & set(df.columns)
        assert len(overlap_cols) == 0, f"Overlap of columns: {overlap_cols}"
        df_snp_info = pd.merge(df_snp_info, df, left_index=True, right_index=True)
        df_snp_info.to_csv(path, sep="\t", float_format="%.8g")


def append_snp_info(
    pfile: str,
    out: str = None,
    info: List[str] = ["LANC_FREQ", "FREQ"],
):
    """
    Append information to .pvar file. Currently, 3 statistics are supported:
    (1) ancestry-specific frequency (2) ancestry-specific haplotype count (3) total
    allele frequency is supported. Please raise an issue if you need other statistics.

    Parameters
    ----------
    pfile : str
        Path to the .pvar file.
    out : str
        Path to the output file. If specified, the output file will be WRITTEN to
        this path. Otherwise, the output file will be appended to the <pfile>.snp_info
        file.
    info : List[str]
        List of information to append. Currently supported:

        * "LANC_FREQ": ancestry-specific allele frequency of each SNP. For example, for a two-way admixture population, `LANC_FREQ1` indicates the frequency of alternate allele in the first ancestry. `LANC_NHAPLO1` will also be added, indicating the number of haplotypes in the first ancestry.
        * "FREQ": allele frequency of each SNP regardless of ancestry.

    Examples
    --------
    .. code-block:: bash
        
        # toy-admix.snp_info will be created containing LANC_FREQ[1-n_anc], LANC_NHAPLO[1-n_anc], FREQ
        admix append-snp-info \\
            --pfile toy-admix \\
            --out toy-admix.snp_info
         
    """
    log_params("append-snp-info", locals())

    dset = admix.io.read_dataset(pfile)

    df_info: pd.DataFrame = []
    if "LANC_FREQ" in info:
        af = dset.af_per_anc()
        nhaplo = dset.nhaplo_per_anc()

        df_af = pd.DataFrame(
            af,
            columns=[f"LANC_FREQ{i + 1}" for i in range(af.shape[1])],
            index=dset.snp.index,
        )
        df_info.append(df_af)

        df_nhaplo = pd.DataFrame(
            nhaplo,
            columns=[f"LANC_NHAPLO{i + 1}" for i in range(nhaplo.shape[1])],
            index=dset.snp.index,
        )
        df_info.append(df_nhaplo)

    if "FREQ" in info:
        df_freq = dapgen.freq(pfile + ".pgen", memory=8)
        assert np.all(df_freq.ID.values == dset.snp.index)
        df_freq = pd.DataFrame(
            df_freq["ALT_FREQS"].values,
            columns=["FREQ"],
            index=dset.snp.index,
        )
        df_info.append(df_freq)

    df_info = pd.concat(df_info, axis=1)

    if out is None:
        out = pfile + ".snp_info"

    _append_to_file(out, df_info)
