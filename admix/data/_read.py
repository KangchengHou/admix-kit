import pandas as pd
import numpy as np
import re
from smart_open import open


def read_vcf(path: str, region: str = None):
    """read vcf file and form xarray.Dataset

    Parameters
    ----------
    path : str
        path to vcf file
    region : str, optional
        region to read, passed to scikit-allel, by default None
    """
    import allel
    import xarray as xr

    vcf = allel.read_vcf(
        path, region=region, fields=["samples", "calldata/GT", "variants/*"]
    )
    gt = vcf["calldata/GT"]
    assert (gt == -1).sum() == 0
    dset = xr.Dataset(
        data_vars={
            "geno": (("indiv", "snp", "ploidy"), np.swapaxes(gt, 0, 1)),
        },
        coords={
            "snp": vcf["variants/ID"].astype(str),
            "indiv": vcf["samples"].astype(str),
            "CHROM@snp": ("snp", vcf["variants/CHROM"].astype(int)),
            "POS@snp": ("snp", vcf["variants/POS"].astype(int)),
            "REF@snp": ("snp", vcf["variants/REF"].astype(str)),
            "ALT@snp": ("snp", vcf["variants/ALT"][:, 0].astype(str)),
            "R2@snp": ("snp", vcf["variants/R2"].astype(float)),
            "MAF@snp": ("snp", vcf["variants/MAF"].astype(float)),
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


def write_digit_mat(path, mat):
    """
    Read a matrix of integer with [0-9], and with no delimiter.

    Args
    ----

    """
    np.savetxt(path, mat, fmt="%d", delimiter="")
