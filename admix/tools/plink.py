from typing import List
from ._utils import get_dependency
import subprocess
import numpy as np
from typing import Union
import pandas as pd


def run(cmd: str):
    """Shortcut for running plink commands

    Parameters
    ----------
    cmd : str
        plink command
    """
    bin_path = get_dependency("plink")
    subprocess.check_call(f"{bin_path} {cmd}", shell=True)


def plink_merge(bfiles: List[str], out: str):
    """Shortcut for merging a list of bfiles

    Parameters
    ----------
    bfiles : List[str]
        List of bfile paths
    out : str
        out prefix
    """

    assert len(bfiles) >= 2
    merge_list = out + ".merge_list"
    np.savetxt(merge_list, bfiles[1:], fmt="%s", delimiter="\n")
    cmd = f"--bfile {bfiles[0]} --merge-list {merge_list} --keep-allele-order --make-bed --out {out}"
    run(cmd)


def read_bim(bim: Union[str, List[str]]):
    if isinstance(bim, list):
        return pd.concat(
            [
                pd.read_csv(
                    b,
                    header=None,
                    delim_whitespace=True,
                    names=["CHR", "SNP", "CM", "POS", "A1", "A2"],
                )
                for b in bim
            ]
        ).reset_index(drop=True)
    elif isinstance(bim, str):
        return pd.read_csv(
            bim,
            header=None,
            delim_whitespace=True,
            names=["CHR", "SNP", "CM", "POS", "A1", "A2"],
        )
    else:
        raise ValueError("`bim` should either be a string or a list of string")


def read_fam(fam: str):
    return pd.read_csv(
        fam,
        header=None,
        delim_whitespace=True,
        usecols=[0, 1],
        names=["FID", "IID"],
    ).astype(str)
