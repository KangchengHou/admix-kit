# interface to external tools
from typing import List, Union
import numpy as np
from numpy.lib.arraysetops import isin
import pandas as pd
import xarray as xr
import tempfile
import urllib
import subprocess
import tempfile
from os.path import join

##################
# common utilities
##################
def has_dependency(name):
    return (
        subprocess.call(
            "which " + name,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        == 0
    )


########
# gcta
########

#######
# plink
#######
def plink_merge(bfiles: List[str], out: str):
    """Shortcut for merging a list of bfiles

    Parameters
    ----------
    bfiles : List[str]
        List of bfile paths
    out : str
        out prefix
    """
    assert has_dependency("plink"), "plink should be in $PATH"

    assert len(bfiles) >= 2
    merge_list = out + ".merge_list"
    np.savetxt(merge_list, bfiles[1:], fmt="%s", delimiter="\n")
    cmd = f"plink --bfile {bfiles[0]} --merge-list {merge_list} --keep-allele-order --make-bed --out {out}"
    subprocess.check_call(cmd, shell=True)


def plink_read_bim(bim: Union[str, List[str]]):
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


def plink_read_fam(fam: str):
    return pd.read_csv(
        fam,
        header=None,
        delim_whitespace=True,
        usecols=[0, 1],
        names=["FID", "IID"],
    )


########################################################################################
############################ miscellaneous #############################################
########################################################################################


def lift_over(chrom_pos: np.ndarray, chain: str, verbose: bool = False):

    """Lift over between genome assembly
    Download appropriate chain file from
    hg38 -> hg19 http://hgdownload.cse.ucsc.edu/goldenpath/hg38/liftOver/hg38ToHg19.over.chain.gz
    hg19 -> hg38 http://hgdownload.cse.ucsc.edu/goldenpath/hg19/liftOver/hg19ToHg38.over.chain.gz
    Download executable from http://hgdownload.soe.ucsc.edu/admin/exe/
    (1) chmod +x liftOver
    (2) include to the path, so the program can call liftOver

    Parameters
    ----------
    chrom_pos: np.ndarray
        2-column matrix where the 1st column are the chromosomes, without `chr`
        and the 2nd column are the positions
    chain : str
        chain file

    Returns
    -------
    np.ndarray
    SNP positions after the liftOver, unmapped SNPs are returned as -1
    """
    assert has_dependency("liftOver"), "liftOver should be in $PATH"

    assert chain in [
        "hg38->hg19",
        "hg19->hg38",
    ], "Currently only these two chain files are supported"
    url_dict = {
        "hg38->hg19": "http://hgdownload.cse.ucsc.edu/goldenpath/hg38/liftOver/hg38ToHg19.over.chain.gz",
        "hg19->hg38": "http://hgdownload.cse.ucsc.edu/goldenpath/hg19/liftOver/hg19ToHg38.over.chain.gz",
    }
    df_old = pd.DataFrame(chrom_pos)
    df_old[0] = df_old[0].apply(lambda x: "chr" + str(x))
    df_old[2] = df_old[1] + 1
    df_old[3] = df_old[0] + ":" + df_old[1].astype(str)

    tmp_dir = tempfile.TemporaryDirectory()

    old_file = join(tmp_dir.name, "old.bed")
    chain_file = join(tmp_dir.name, "chain.gz")
    new_file = join(tmp_dir.name, "new.bed")
    unmapped_file = join(tmp_dir.name, "unmapped.txt")

    urllib.request.urlretrieve(url_dict[chain], chain_file)
    df_old.to_csv(old_file, sep="\t", index=False, header=False)
    cmd = f"liftOver {old_file} {chain_file} {new_file} {unmapped_file}"
    subprocess.check_call(cmd, shell=True)
    df_new = pd.read_csv(new_file, sep="\t", header=None)
    df_merged = df_old.set_index(3).join(
        df_new.set_index(3), lsuffix="_old", rsuffix="_new"
    )
    df_unmapped = pd.read_csv(unmapped_file, comment="#", sep="\t", header=None)
    if verbose:
        print("Unmapped SNPs:")
        print(df_unmapped)
    tmp_dir.cleanup()

    return df_merged["1_new"].fillna(-1).astype(int).values
