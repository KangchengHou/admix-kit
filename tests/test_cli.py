"""
End-to-end tests for the admix command line interfaces.
"""
import admix
import tempfile
from admix._utils import cd
import subprocess
import pandas as pd
import numpy as np


def test_assoc_quant():
    """
    Test that the CLI is consistent with the python API.
    admix assoc-quant \
        --pfile toy-admix \
        --pheno toy-admix.indiv_info \
        --pheno-col PHENO \
        --covar toy-admix.indiv_info \
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

    with tempfile.TemporaryDirectory() as tmp_dir:
        with cd(tmp_dir):
            cmds = [
                "admix assoc-quant",
                f"--pfile {data_dir}/toy-admix",
                f"--pheno {data_dir}/toy-admix.indiv_info",
                "--pheno-col PHENO",
                "--method ATT,TRACTOR",
                "--out toy-admix.assoc",
            ]
            subprocess.check_call(" ".join(cmds), shell=True)
            df_assoc = pd.read_csv("toy-admix.assoc", sep="\t", index_col=0)
    print(df_assoc)
    assert np.allclose(
        df_assoc.loc[dset.snp.index, "ATT"],
        dset.snp["ATT"],
    )
    assert np.allclose(df_assoc.loc[dset.snp.index, "TRACTOR"], dset.snp["TRACTOR"])

    # check that after including covariates, the results are not the same
    with tempfile.TemporaryDirectory() as tmp_dir:
        with cd(tmp_dir):
            dset.indiv[["PC1", "PC2"]].to_csv("toy-admix.covar", sep="\t")
            print(dset.indiv[["PC1", "PC2"]])
            cmds = [
                "admix assoc-quant",
                f"--pfile {data_dir}/toy-admix",
                f"--pheno {data_dir}/toy-admix.indiv_info",
                "--pheno-col PHENO",
                f"--covar toy-admix.covar",
                "--method ATT,TRACTOR",
                "--out toy-admix.assoc",
            ]
            subprocess.check_call(" ".join(cmds), shell=True)
            df_assoc = pd.read_csv("toy-admix.assoc", sep="\t", index_col=0)
    print(df_assoc)
    assert not np.allclose(
        df_assoc.loc[dset.snp.index, "ATT"],
        dset.snp["ATT"],
    )
    assert not np.allclose(
        df_assoc.loc[dset.snp.index, "TRACTOR"],
        dset.snp["TRACTOR"],
    )


def test_consistent():
    """
    Test that the CLI is consistent with the API.
    """
    pass
