from numpy import (
    asarray,
    float32,
    float64,
    fromfile,
    int64,
    tril,
    tril_indices_from,
    zeros,
)
from pandas import read_csv
import numpy as np
import re
import dask.array as da
from typing import List, Optional
import xarray as xr
import admix
import dapgen
import os
import pandas as pd


def read_lanc(path: str) -> admix.data.Lanc:
    """Read local ancestry with .lanc format

    Parameters
    ----------
    """
    lanc = admix.data.Lanc(path)
    return lanc


def read_dataset(
    pfile: str,
    lanc_file: str = None,
    snp_info_file: str = None,
    indiv_info_file: str = None,
    n_anc: int = None,
    snp_chunk: int = 1024,
) -> admix.Dataset:
    """
    TODO: support multiple pfile, such as data/chr*
    TODO: support only reading a subset of the individuals

    Read a dataset from a directory.

    pfile.snp_info will also be read and combined with .pvar

    Parameters
    ----------
    pfile: str
        PLINK2 file prefix
    lanc_file: str
        local ancestry file, if not provided, `read_dataset` will attempt to find it
        with <pfile>.lanc
    snp_info_file: str
        SNP info file, if not provided, `read_dataset` will attempt to find it
        with <pfile>.snp_info
    indiv_info_file: str
        individual info file, if not provided, `read_dataset` will attempt to find it
        with <pfile>.indiv_info
    n_anc: int
        number of ancestries, if not provided, `read_dataset` will attempt to infer from
        the local ancestry file
    snp_chunk: int
        chunk size for reading the SNP info file (default: 1024)

    Returns
    -------
    Dataset
    """

    # infer local ancestry file
    if lanc_file is None:
        if os.path.exists(pfile + ".lanc"):
            lanc_file = pfile + ".lanc"
    if lanc_file is not None:
        lanc = admix.io.read_lanc(lanc_file).dask(snp_chunk=snp_chunk)
    else:
        lanc = None

    # infer SNP info file
    if snp_info_file is None:
        if os.path.exists(pfile + ".snp_info"):
            snp_info_file = pfile + ".snp_info"

    if indiv_info_file is None:
        if os.path.exists(pfile + ".indiv_info"):
            indiv_info_file = pfile + ".indiv_info"

    geno, pvar, psam = dapgen.read_pfile(pfile, phase=True, snp_chunk=snp_chunk)

    dset = admix.Dataset(geno=geno, lanc=lanc, snp=pvar, indiv=psam, n_anc=n_anc)

    if snp_info_file is not None:
        df_snp_info = pd.read_csv(snp_info_file, index_col=0, sep="\t")
        assert (
            len(set(dset.snp.columns) & set(df_snp_info.columns)) == 0
        ), "SNP info file columns must not overlap with dset columns"
        dset._snp = pd.merge(
            dset.snp,
            df_snp_info.reindex(dset.snp.index),
            left_index=True,
            right_index=True,
        )

    if indiv_info_file is not None:
        df_indiv_info = pd.read_csv(
            indiv_info_file,
            index_col=0,
            sep="\t",
            low_memory=False,
        )
        assert (
            len(set(dset.indiv.columns) & set(df_indiv_info.columns)) == 0
        ), "there should be no intersection between dest.indiv.columns and indiv_info.columns"
        dset._indiv = pd.merge(
            dset.indiv,
            df_indiv_info.reindex(dset.indiv.index),
            left_index=True,
            right_index=True,
        )
    return dset


def read_vcf(path: str, region: str = None, samples: List[str] = None) -> Optional[xr.Dataset]:
    """read vcf file and form xarray.Dataset

    Parameters
    ----------
    path : str
        path to vcf file
    region : str, optional
        region to read, passed to scikit-allel, by default None

    Returns
    -------
    xarray.Dataset
        xarray.Dataset, if no snps in region, return None
    """
    import allel
    import xarray as xr

    vcf = allel.read_vcf(
        path,
        region=region,
        samples=samples,
        fields=["samples", "calldata/GT", "variants/*"],
    )
    if vcf is None:
        return None

    gt = vcf["calldata/GT"]
    assert (gt == -1).sum() == 0

    # used to convert chromosome to int
    chrom_format_func = np.vectorize(lambda x: int(x.replace("chr", "")))
    dset = xr.Dataset(
        data_vars={
            "geno": (("indiv", "snp", "ploidy"), da.from_array(np.swapaxes(gt, 0, 1))),
        },
        coords={
            "snp": vcf["variants/ID"].astype(str),
            "indiv": vcf["samples"].astype(str),
            "CHROM": (
                "snp",
                chrom_format_func(vcf["variants/CHROM"]),
            ),
            "POS": ("snp", vcf["variants/POS"].astype(int)),
            "REF": ("snp", vcf["variants/REF"].astype(str)),
            "ALT": ("snp", vcf["variants/ALT"][:, 0].astype(str)),
            "R2": ("snp", vcf["variants/R2"].astype(float)),
            "MAF": ("snp", vcf["variants/MAF"].astype(float)),
        },
    )
    return dset


def read_digit_mat(path: str, filter_non_numeric: bool = False, nrows: int = None):
    """
    Read a matrix of integer with [0-9], and with no delimiter.

    Parameters
    ----------
    path : str
        path to the matrix file
    filter_non_numeric : bool, optional
        whether to filter out non-numeric characters, by default False
    nrows : int, optional
        number of rows to read, by default None

    Returns
    -------
    np.ndarray
        matrix of integer
    """
    if nrows is None:
        if filter_non_numeric:
            with open(path) as f:
                mat = np.array(
                    [
                        np.array([int(c) for c in re.sub("[^0-9]", "", line.strip())])
                        for line in f.readlines()
                    ],
                    dtype=np.int8,
                )
        else:
            with open(path) as f:
                mat = np.array(
                    [np.array([int(c) for c in line.strip()]) for line in f.readlines()],
                    dtype=np.int8,
                )
    else:
        assert filter_non_numeric is False
        mat = np.array(
            [
                np.array([int(c) for c in line.item()])
                for line in pd.read_csv(path, nrows=nrows, header=None).values
            ],
            dtype=np.int8,
        )
    return mat


def read_gcta_grm(file_prefix) -> dict:
    """
    Reads the GRM from a GCTA formated file.

    Parameters
    ----------
    file_prefix : str
        The prefix of the GRM to be read.

    Returns
    -------
    dict
        A dictionary with the GRM values.
        - grm: GRM matrix
        - df_id: ids of the individuals
        - n_snps: number of SNP

    """

    bin_file = file_prefix + ".grm.bin"
    N_file = file_prefix + ".grm.N.bin"
    id_file = file_prefix + ".grm.id"

    df_id = read_csv(id_file, sep="\t", header=None, names=["sample_0", "sample_1"])
    n = df_id.shape[0]
    k = asarray(fromfile(bin_file, dtype=float32), float64)
    n_snps = asarray(fromfile(N_file, dtype=float32), int64)

    K = zeros((n, n))
    K[tril_indices_from(K)] = k
    K = K + tril(K, -1).T
    return {
        "grm": K,
        "df_id": df_id,
        "n_snps": n_snps,
    }


def read_rfmix(
    path: str,
    df_snp: pd.DataFrame,
    df_indiv: pd.DataFrame,
):
    """
    Assign local ancestry to a dataset. 2-way, 3-way admixture are all supported.

    Parameters
    ----------
    lanc_file: str
        Path to local ancestry data.
    geno: xr.DataArray
        genotype matrix
    df_snp: pd.DataFrame
        SNP data frames

    Returns
    -------
    lanc: da.Array
        Local ancestry array
    """

    # assign local ancestry
    df_rfmix = pd.read_csv(path, sep="\t", skiprows=1)
    assert np.unique(df_rfmix["#chm"]).shape[0] == 1, "rfmix can only contain one chromosome"
    assert np.unique(df_snp["CHROM"]).shape[0] == 1, "df_snp can only contain one chromosome"
    assert set(df_rfmix["#chm"].values) == set(
        df_snp["CHROM"].values
    ), "rfmix and df_snp must contain the same chromosome"
    # read the 2 haplotype, the value of those haplotype corresponds to the local ancestry
    lanc0 = df_rfmix.loc[:, df_rfmix.columns.str.endswith(".0")].rename(columns=lambda x: x[:-2])
    lanc1 = df_rfmix.loc[:, df_rfmix.columns.str.endswith(".1")].rename(columns=lambda x: x[:-2])
    assert (
        np.any([col.endswith(".2") for col in df_rfmix.columns]) == False
    ), "There are columns with .2 in the rfmix file (not expected), please raise an issue"

    lanc = lanc0.astype(str) + lanc1.astype(str)

    df_rfmix_info = df_rfmix.iloc[:, 0:3].copy()
    # extend local ancestry to two ends of chromosomes if necessary
    df_rfmix_info.loc[0, "spos"] = min(df_snp["POS"][0] - 1, df_rfmix_info.loc[0, "spos"])
    df_rfmix_info.loc[len(df_rfmix_info) - 1, "epos"] = max(
        df_snp["POS"][-1] + 1, df_rfmix_info.loc[len(df_rfmix_info) - 1, "epos"]
    )

    assert np.all(df_indiv.index == lanc.columns)

    n_indiv = len(df_indiv)
    n_snp = len(df_snp)

    rfmix_break_list = np.zeros(df_rfmix_info.shape[0], dtype=int)
    # [start, stop) of SNPs for each rfmix break points
    # find the RFmix break points in coordinates of SNP location
    chunk_stop = 0
    for chunk_i, chunk in df_rfmix_info.iterrows():
        chunk_mask = np.where((chunk.spos <= df_snp["POS"]) & (df_snp["POS"] < chunk.epos))[0]
        if len(chunk_mask) > 0:
            chunk_stop = chunk_mask[-1]
        else:
            # use the previous chunk stop if no SNP in this chunk
            admix.logger.warning(f"No SNP found in {chunk_i}-th region of RFmix")
            chunk_stop = chunk_stop
        rfmix_break_list[chunk_i] = chunk_stop

    # find break points in the data
    chunk_pos, indiv_pos = np.where(lanc.iloc[1:, :].values != lanc.iloc[:-1, :].values)
    # convert to SNP positions
    snp_pos = rfmix_break_list[chunk_pos]
    values = lanc.values[chunk_pos, indiv_pos]

    # append values at the end of the chromosomes
    snp_pos = np.concatenate([snp_pos, [n_snp - 1] * n_indiv])
    indiv_pos = np.concatenate([indiv_pos, np.arange(n_indiv)])
    values = np.concatenate([values, lanc.iloc[-1].values])

    # snp_pos, indiv_pos, values are now triples of break points

    break_list = []
    value_list = []
    # convert to .lanc format
    for indiv_i in range(n_indiv):
        indiv_mask = indiv_pos == indiv_i
        # +1 because .lanc denote the [start, stop) of the break points
        indiv_snp_pos, unique_mask = np.unique(snp_pos[indiv_mask] + 1, return_index=True)
        indiv_values = values[indiv_mask][unique_mask]
        break_list.append(indiv_snp_pos.tolist())
        value_list.append(indiv_values.tolist())

    return admix.data.Lanc(breaks=break_list, values=value_list)


def read_joint_pca(pca_prefix: str, ref_pfile: str):
    """
    Read joint PCA results.

    Parameters
    ----------
    ref_pfile : str
        reference panel pfile prefix
    pca_prefix : str
        joint pca results prefix. {pca_prefix}.eigenvec, {pca_prefix}.eigenval
        will be read

    Returns
    -------
    df_pca : pd.DataFrame
        PCA results
    eigenval : np.ndarray
        eigenvalues
    """
    df_pca = (
        pd.read_csv(f"{pca_prefix}.eigenvec", delim_whitespace=True)
        .set_index("IID")
        .drop(columns=["#FID"])
    )

    with open(f"{pca_prefix}.eigenval") as f:
        eigenval = np.array([float(l.strip()) for l in f.readlines()])

    df_pop = pd.read_csv(ref_pfile + ".psam", delim_whitespace=True, index_col=0)
    assert (
        "SuperPop" in df_pop.columns and "Population" in df_pop.columns
    ), f"SuperPop and Population columns are required in the reference panel {ref_pfile}.psam file"
    
    df_pca["SUPERPOP"] = df_pop["SuperPop"].reindex(df_pca.index)
    df_pca["POP"] = df_pop["Population"].reindex(df_pca.index)
    # make sure SAMPLE is not in SUPERPOP or POP
    assert "SAMPLE" not in df_pca["SUPERPOP"].unique()
    assert "SAMPLE" not in df_pca["POP"].unique()
    df_pca["SUPERPOP"] = df_pca["SUPERPOP"].fillna("SAMPLE")
    df_pca["POP"] = df_pca["POP"].fillna("SAMPLE")

    # reorder SAMPLE to the bottom of the data frame
    df_pca = pd.concat([df_pca[df_pca.POP != "SAMPLE"], df_pca[df_pca.POP == "SAMPLE"]])
    return df_pca, eigenval