"""
End-to-end tests for the admix command line interfaces.
"""
import admix
import tempfile
from admix.utils import cd
import subprocess
import pandas as pd
import numpy as np


def test_assoc_quant():
    """
    Test that the CLI is consistent with the python API.
    admix assoc \
        --pfile toy-admix \
        --pheno toy-admix.indiv_info \
        --method ATT,TRACTOR \
        --out toy-admix.assoc

    """
    dset = admix.dataset.load_toy_admix()

    # how two implementations cope with zero-freq SNPs are a bit different for now.
    # subset the dataset to avoid this for now
    test_eq_idx = (
        (dset.snp.FREQ1.between(0.01, 0.99)) & (dset.snp.FREQ2.between(0.01, 0.99))
    ).values
    dset = dset[test_eq_idx]
    data_dir = admix.dataset.get_test_data_dir()
    df_pheno = pd.read_csv(f"{data_dir}/toy-admix.indiv_info", sep="\t", index_col=0)
    with tempfile.TemporaryDirectory() as tmp_dir:
        with cd(tmp_dir):
            df_pheno[["PHENO"]].to_csv("pheno.txt", sep="\t", na_rep="NA")
            cmds = [
                "admix assoc",
                f"--pfile {data_dir}/toy-admix",
                f"--pheno pheno.txt",
                "--method ATT,TRACTOR",
                "--out toy-admix",
                "--family quant",
            ]
            subprocess.check_call(" ".join(cmds), shell=True)
            for m in ["ATT", "TRACTOR"]:
                df_assoc = pd.read_csv(f"toy-admix.{m}.assoc", sep="\t", index_col=0)
                assert np.allclose(
                    df_assoc.loc[dset.snp.index, "P"],
                    dset.snp[m],
                )

    # check that after including covariates, the results are not the same
    with tempfile.TemporaryDirectory() as tmp_dir:
        with cd(tmp_dir):
            df_pheno[["PHENO", "PC1", "PC2"]].to_csv("pheno.txt", sep="\t", na_rep="NA")
            cmds = [
                "admix assoc",
                f"--pfile {data_dir}/toy-admix",
                f"--pheno pheno.txt",
                "--method ATT,TRACTOR",
                "--out toy-admix",
                "--family quant",
            ]
            subprocess.check_call(" ".join(cmds), shell=True)
            for m in ["ATT", "TRACTOR"]:
                df_assoc = pd.read_csv(f"toy-admix.{m}.assoc", sep="\t", index_col=0)
                assert not np.allclose(df_assoc.loc[dset.snp.index, "P"], dset.snp[m])
