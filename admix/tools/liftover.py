from . import get_dependency
import numpy as np
import tempfile
import urllib.request
import subprocess
from os.path import join
import pandas as pd


def run(df_chrom_pos: pd.DataFrame, chain: str, verbose: bool = False):

    """Lift over between genome assembly
    Download appropriate chain file from
    hg38->hg19 http://hgdownload.cse.ucsc.edu/goldenpath/hg38/liftOver/hg38ToHg19.over.chain.gz
    hg19->hg38 http://hgdownload.cse.ucsc.edu/goldenpath/hg19/liftOver/hg19ToHg38.over.chain.gz
    Download executable from http://hgdownload.soe.ucsc.edu/admin/exe/
    (1) chmod +x liftOver
    (2) include to the path, so the program can call liftOver

    Parameters
    ----------
    df_chrom_pos: pd.DataFrame
        where the 1st column are the chromosomes, without `chr`
        and the 2nd column are the positions
    chain : str
        chain file

    Returns
    -------
    index of df_chrom_pos and the corresponding positions
    SNP positions after the liftOver, unmapped SNPs are returned as -1
    """

    df_chrom_pos = df_chrom_pos.copy()
    df_chrom_pos.columns = ["CHROM", "POS"]
    bin_path = get_dependency("liftOver")

    assert chain in [
        "hg38->hg19",
        "hg19->hg38",
    ], "Currently only these two chain files are supported"
    url_dict = {
        "hg38->hg19": "http://hgdownload.cse.ucsc.edu/goldenpath/hg38/liftOver/hg38ToHg19.over.chain.gz",
        "hg19->hg38": "http://hgdownload.cse.ucsc.edu/goldenpath/hg19/liftOver/hg19ToHg38.over.chain.gz",
    }
    df_chrom_pos = pd.DataFrame(
        {
            "CHROM": "chr" + df_chrom_pos["CHROM"].astype(str),
            "POS0": df_chrom_pos["POS"] - 1,
            "POS": df_chrom_pos["POS"],
            "ID": df_chrom_pos.index.values.astype(str),
        }
    )

    tmp_dir = tempfile.TemporaryDirectory()
    old_file = join(tmp_dir.name, "old.bed")
    chain_file = join(tmp_dir.name, "chain.gz")
    new_file = join(tmp_dir.name, "new.bed")
    unmapped_file = join(tmp_dir.name, "unmapped.txt")

    urllib.request.urlretrieve(url_dict[chain], chain_file)
    df_chrom_pos.to_csv(old_file, sep="\t", index=False, header=False)
    cmd = f"{bin_path} {old_file} {chain_file} {new_file} {unmapped_file}"
    subprocess.check_call(cmd, shell=True)

    # read the mapped data frame, SNPs with ambiguous mapping are dropped
    df_new = pd.read_csv(
        new_file, sep="\t", header=None, names=["CHROM", "POS0", "POS", "ID"]
    ).drop_duplicates(subset=["ID"], keep=False)
    # count number of lines starting with a #
    n_unmapped = sum(1 for line in open(unmapped_file) if line.startswith("#"))
    print(f"{n_unmapped} un-mapped variants")

    tmp_dir.cleanup()

    # ambiguous mapping or unmapped SNPs are -1, return
    df_ret = (
        df_new.set_index("ID")
        .reindex(df_chrom_pos.index.values)[["POS"]]
        .fillna(-1)
        .astype(int)
    )

    df_ret.index.name = df_chrom_pos.index.name
    return df_ret