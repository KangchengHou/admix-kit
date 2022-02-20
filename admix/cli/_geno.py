import os
from typing import List
import admix
import pandas as pd
import dapgen


def _append_pvar(pvar_path: str, df: pd.DataFrame):
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
    df_pvar, header = dapgen.read_pvar(pvar_path, return_header=True)
    assert df_pvar.index.equals(df.index)
    # no overlap of columns
    overlap_cols = set(df_pvar.columns) & set(df.columns)
    assert len(overlap_cols) == 0, f"Overlap of columns: {overlap_cols}"
    df_pvar = pd.merge(df_pvar, df, left_index=True, right_index=True)
    dapgen.write_pvar(pvar_path, df_pvar, header)


def append_snp_info(
    pfile: str,
    info: List[str] = ["LANC_FREQ"],
):
    """
    Append information to .pvar file. Currently only the ancestry-specific frequency
    of the SNP is supported.

    Parameters
    ----------
    pfile : str
        Path to the .pvar file.
    info : List[str]
        List of information to append. Currently supported:
        - "LANC_FREQ": ancestry-specific allele frequency of each SNP. For example,
            for a two-way admixture individual, LANC_FREQ1 indicates the frequency of
            alternate allele in the first ancestry.
    """
    dset = admix.io.read_dataset(pfile)

    df_info = []
    if "LANC_FREQ" in info:
        af = dset.af_per_anc()
        df_af = pd.DataFrame(
            af,
            columns=[f"LANC_FREQ{i + 1}" for i in range(af.shape[1])],
            index=dset.snp.index,
        )
        df_info.append(df_af)

    df_info = pd.concat(df_info, axis=1)
    _append_pvar(pfile + ".pvar", df_info)