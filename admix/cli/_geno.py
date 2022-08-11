import os
from typing import List, Dict
import admix
import pandas as pd
import dapgen
import numpy as np
import dask.array as da
from tqdm import tqdm
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


def da_sum(mat: da.Array):
    """Summation over a dask array to get individual-level sum

    Parameters
    ----------
    mat : da.Array
        dask array to sum over
    """
    chunks = mat.chunks[0]
    indices = np.insert(np.cumsum(chunks), 0, 0)
    mat_sum = np.zeros(mat.shape[1])
    for i in range(len(indices) - 1):
        start, stop = indices[i], indices[i + 1]
        mat_sum += mat[start:stop, :].sum(axis=0).compute()
    return mat_sum


def calc_pgs(
    plink_path: str,
    weights_path: str,
    out: str,
    weight_col: str = "WEIGHT",
):
    """Calculate PGS from a weight file and a PLINK file.

    Parameters
    ----------
    plink_path : str
        Path to plink files. Format examples:
          * /path/to/chr21.pgen
          * /path/to/genotype/directory
          * /path/to/file_list.txt # file_list.txt contains rows of file names
    weights_path : str
        path to PGS weights, containing CHROM, SNP, REF, ALT, WEIGHT columns
    out : str
        prefix of the output files.
    weight_col : str, optional
        column in 'weights_path' representing the weight, by default "WEIGHT"
    """
    log_params("calc-pgs", locals())
    df_weights = pd.read_csv(weights_path, sep="\t")
    pgen_files = dapgen.parse_plink_path(plink_path)
    if not isinstance(pgen_files, list):
        pgen_files = [pgen_files]
    plink_prefix_list = [pgen.rsplit(".", 1)[0] for pgen in pgen_files]

    df_snp_info: Dict = {"PLINK_SNP": [], "WEIGHTS_SNP": [], "WEIGHTS": []}
    all_partial_pgs = 0.0
    all_lanc = 0.0
    indiv_list = None
    all_n_snp = 0
    for plink_prefix in plink_prefix_list:
        dset = admix.io.read_dataset(plink_prefix)
        assert dset.n_anc == 2, "Only 2-way admixture is currently supported"
        if indiv_list is None:
            indiv_list = dset.indiv.index
        else:
            assert indiv_list.equals(
                dset.indiv.index
            ), f"Indiv list in {plink_prefix} does not match with previous ones."
        dset_idx, wgt_idx, flip = dapgen.align_snp(
            df1=dset.snp[["CHROM", "POS", "REF", "ALT"]], df2=df_weights
        )
        dset_subset = dset[dset_idx]
        admix.logger.info(f"# matched SNPs={len(wgt_idx)} for dset={plink_prefix}.")
        tmp_df_weights = df_weights[[weight_col]].loc[wgt_idx, :] * flip.reshape(-1, 1)
        tmp_df_weights.index = dset_idx

        partial_pgs = admix.data.calc_pgs(
            dset=dset_subset, df_weights=tmp_df_weights, method="partial"
        )

        df_snp_info["PLINK_SNP"].extend(dset_idx)
        df_snp_info["WEIGHTS_SNP"].extend(wgt_idx)
        df_snp_info["WEIGHTS"].extend(df_weights.loc[wgt_idx, weight_col].values * flip)
        all_partial_pgs += partial_pgs
        all_lanc += da_sum(dset_subset.lanc.sum(axis=2))
        all_n_snp += dset_subset.n_snp

    pd.DataFrame(df_snp_info).to_csv(out + ".snp_info.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "indiv": dset.indiv.index,
            "PGS1": all_partial_pgs[:, 0],
            "PGS2": all_partial_pgs[:, 1],
            "PROP1": 1 - all_lanc / (all_n_snp * 2),
            "PROP2": all_lanc / (all_n_snp * 2),
        }
    ).to_csv(out + ".pgs.tsv", sep="\t", index=False)
    admix.logger.info(f"SNP information saved to {out}.snp_info.tsv")
    admix.logger.info(f"PGS saved to {out}.pgs.tsv")


def calc_partial_pgs(
    plink_path: str,
    weights_path: str,
    ref_plink_path: str,
    ref_pops: List[str],
    out: str,
    weight_col: str = "WEIGHT",
    ref_pop_col: str = "Population",
):
    """Calculate PGS from a weight file and a PLINK file.

    Parameters
    ----------
    plink_path : str
        Path to plink files. Format examples:
          * /path/to/chr21.pgen
          * /path/to/genotype/directory
          * /path/to/file_list.txt # file_list.txt contains rows of file names
    weights_path : str
        path to PGS weights, containing CHROM, SNP, REF, ALT, WEIGHT columns
    ref_plink_path : str
        path to reference plink files.
    ref_pops: list of str
        list of populations in reference plink files.
    out : str
        prefix of the output files.
    weight_col : str, optional
        column in 'weights_path' representing the weight, by default "WEIGHT"
    """
    # TODO: more hints for the user
    log_params("calc-partial-pgs", locals())

    CHECK_COLS = ["CHROM", "POS", "REF", "ALT"]

    # read input data & basic checks
    pgen_files = dapgen.parse_plink_path(plink_path)
    if not isinstance(pgen_files, list):
        pgen_files = [pgen_files]
    plink_prefix_list = [pgen.rsplit(".", 1)[0] for pgen in pgen_files]

    # scoring weights
    df_weights = pd.read_csv(weights_path, sep="\t")
    df_weights = df_weights[CHECK_COLS + [weight_col]].copy()

    # reference data
    dset_ref = admix.io.read_dataset(ref_plink_path)
    ref_pop_indiv = {
        pop: dset_ref.indiv.index[dset_ref.indiv[ref_pop_col] == pop].values
        for pop in ref_pops
    }

    total_sample_pgs = 0
    total_ref_pgs = 0
    # iterate through each plink file
    for plink_prefix in tqdm(plink_prefix_list):
        _dset = admix.io.read_dataset(plink_prefix)
        assert _dset.n_anc == 2, "Only 2-way admixture is currently supported"
        # align sample dset and weights
        idx1, idx2, flip = dapgen.align_snp(
            df1=_dset.snp[CHECK_COLS], df2=df_weights[CHECK_COLS]
        )
        _dset = _dset[idx1]
        _df_weights = df_weights.loc[idx2, :].copy()
        _df_weights.index = idx1

        # align sample dset and reference dset
        idx1, idx2, flip = dapgen.align_snp(
            df1=_dset.snp[CHECK_COLS],
            df2=dset_ref.snp[CHECK_COLS],
        )
        _dset = _dset[idx1]
        _df_weights = _df_weights.loc[idx1, :]
        _dset_ref = dset_ref[idx2]
        sample_pgs, ref_pgs = admix.data.calc_partial_pgs(
            dset=_dset,
            df_weights=_df_weights,
            dset_ref=_dset_ref,
            ref_pops=ref_pop_indiv,
        )
        total_sample_pgs += sample_pgs
        total_ref_pgs += ref_pgs

    total_sample_pgs.to_csv(out + ".sample_pgs.tsv", sep="\t")
    total_ref_pgs.to_csv(out + ".ref_pgs.tsv", sep="\t")
    admix.logger.info(f"Sample PGS saved to {out}.sample_pgs.tsv")
    admix.logger.info(f"Reference PGS saved to {out}.ref_pgs.tsv")