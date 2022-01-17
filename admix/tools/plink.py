from typing import List
from ._utils import get_dependency
import subprocess
import numpy as np
from typing import Union
import pandas as pd
import admix
import os
import glob


def run(cmd: str, **kwargs):
    """Shortcut for running plink commands

    Parameters
    ----------
    cmd : str
        plink command
    """
    bin_path = get_dependency("plink")
    add_cmds = [f" --{k.replace('_', '-')} {kwargs[k]}" for k in kwargs]
    cmd += " ".join(add_cmds)

    subprocess.check_call(f"{bin_path} {cmd}", shell=True)


def merge(bfiles: List[str], out: str, **kwargs):
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


def clump(
    bfile: str,
    assoc_path: str,
    out_prefix: str,
    p1: float = 5e-8,
    p2: float = 1e-4,
    r2: float = 0.1,
    kb=3000,
    **kwargs,
):
    """
    Wrapper for plink2 clump

    Parameters
    ----------
    bfile : str
        Path to plink binary file
    assoc_path : str
        Path to association file, assumed to be PLINK2 format, the function will
        perform the necessary conversion
    out_prefix : str
        Prefix for output files
    p1 : float, optional
        p-value threshold for clumping, by default 5e-8
    p2 : float, optional
        p-value threshold for clumping, by default 1e-4
    r2 : float, optional
        r2 threshold for clumping, by default 0.1
    kb : int, optional
        Number of kb to use for clumping, by default 3000
    """
    tmp_prefix = out_prefix + ".admix_plink_clump_tmp"
    # convert plink2 association to plink1 format ID -> SNP
    import shutil

    from_file = open(assoc_path)
    to_file = open(tmp_prefix + ".assoc", "w")
    to_file.writelines(from_file.readline().replace("ID", "SNP"))
    shutil.copyfileobj(from_file, to_file)
    from_file.close()
    to_file.close()
    cmds = [
        f"--bfile {bfile} --clump {tmp_prefix + '.assoc'}",
        f"--clump-p1 {p1} --clump-p2 {p2} --clump-r2 {r2} --clump-kb {kb}",
        f"--out {tmp_prefix}",
    ]

    admix.tools.plink.run(" ".join(cmds), **kwargs)
    if os.path.exists(tmp_prefix + ".clumped"):
        os.rename(tmp_prefix + ".clumped", out_prefix + ".clumped")
    else:
        # no clumped region
        # write a comment to the output file
        with open(out_prefix + ".clumped", "w") as file:
            file.write("# No clumped region")

    for f in glob.glob(tmp_prefix + "*"):
        os.remove(f)
