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
from smart_open import open
import dask.array as da
import dask


def read_plink(path: str):
    """read plink file and form xarray.Dataset

    Parameters
    ----------
    path : str
        path to plink file prefix without .bed/.bim/.fam
    """
    import xarray as xr
    import pandas_plink

    # count number of a0 as dosage, (A1 in usual PLINK bim file)
    plink = pandas_plink.read_plink1_bin(
        f"{path}.bed",
        chunk=pandas_plink.Chunk(nsamples=None, nvariants=1024),
        verbose=False,
        ref="a0",
    )

    dset = xr.Dataset(
        {
            "geno": xr.DataArray(
                data=plink.data,
                coords={
                    "indiv": (plink["fid"] + "_" + plink["iid"]).values.astype(str),
                    "snp": plink["snp"].values.astype(str),
                    "CHROM": ("snp", plink["chrom"].values.astype(int)),
                    "POS": ("snp", plink["pos"].values.astype(int)),
                    "REF": ("snp", plink["a1"].values.astype(str)),
                    "ALT": ("snp", plink["a0"].values.astype(str)),
                },
                dims=["indiv", "snp"],
            )
        }
    )

    return dset


def read_vcf(path: str, region: str = None):
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
        path, region=region, fields=["samples", "calldata/GT", "variants/*"]
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


def read_digit_mat(path, filter_non_numeric=False):
    """
    Read a matrix of integer with [0-9], and with no delimiter.

    Args
    ----

    """
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