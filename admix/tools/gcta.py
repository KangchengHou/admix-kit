import pandas as pd
import numpy as np
import subprocess
from typing import Tuple
import os
from . import get_dependency


def run(cmd: str):
    """Shortcut for running plink commands

    Parameters
    ----------
    cmd : str
        gcta command
    """
    bin_path = get_dependency("gcta64")
    subprocess.check_call(f"{bin_path} {cmd}", shell=True)


def read_grm(file_prefix: str) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """read GCTA grm file

    Parameters
    ----------
    file_prefix : str
        prefix of the grm file, <file_prefix>.grm.bin, <file_prefix>.grm.id,
        <file_prefix>.grm.N.bin should be present

    Returns
    -------
    Tuple[np.ndarray, pd.DataFrame, np.ndarray]
        K: grm matrix
        df_id: id data frame
        N: N matrix
    """
    bin_file = file_prefix + ".grm.bin"
    N_file = file_prefix + ".grm.N.bin"
    id_file = file_prefix + ".grm.id"

    df_id = pd.read_csv(id_file, sep="\t", header=None, names=["sample_0", "sample_1"])
    n = df_id.shape[0]
    k = np.asarray(np.fromfile(bin_file, dtype=np.float32), np.float64)
    n_snps = np.asarray(np.fromfile(N_file, dtype=np.float32), np.int64)

    K = np.zeros((n, n))
    K[np.tril_indices_from(K)] = k
    K = K + np.tril(K, -1).T

    return (K, df_id, n_snps)


def write_grm(file_prefix, K, df_id, n_snps):

    bin_file = file_prefix + ".grm.bin"
    N_file = file_prefix + ".grm.N.bin"
    id_file = file_prefix + ".grm.id"

    # id
    df_id.to_csv(id_file, sep="\t", header=None, index=False)
    # bin
    K[np.tril_indices_from(K)].astype(np.float32).tofile(bin_file)
    # N
    n_snps.astype(np.float32).tofile(N_file)


def reml(
    df_pheno: pd.DataFrame,
    out_prefix: str,
    mgrm_path: str = None,
    grm_path: str = None,
    df_covar: pd.DataFrame = None,
    n_thread: int = 4,
    clean_tmp: bool = True,
):
    """Wrapper for GCTA --reml

    Parameters
    ----------
    mgrm_path : str
        mgrm path to file with a list of grms (without .grm.bin, .grm.id, .grm.N.bin)
    grm_path : str
        grm path prefix (without .grm.bin, .grm.id, .grm.N.bin)
    df_pheno : pd.DataFrame
        phenotype data frame
    out_prefix : str
        output prefix
    """
    assert (mgrm_path is None) != (
        grm_path is None
    ), "Either mgrm_path or grm_path must be provided"
    assert df_pheno.shape[1] == 3
    assert df_pheno.columns[0] == "FID"
    assert df_pheno.columns[1] == "IID"

    pheno_path = out_prefix + ".pheno"
    df_pheno.to_csv(pheno_path, index=False, header=False, sep="\t", na_rep="NA")

    cmds = [
        "--reml --reml-no-lrt --reml-no-constrain",
        f"--pheno {pheno_path}",
        f"--out {out_prefix}",
        f"--thread-num {n_thread}",
    ]
    if mgrm_path is not None:
        cmds.append(f"--mgrm {mgrm_path}")
    if grm_path is not None:
        cmds.append(f"--grm {grm_path}")

    if df_covar is not None:
        covar_path = out_prefix + ".covar"
        df_covar.to_csv(covar_path, index=False, header=False, sep="\t", na_rep="NA")
        cmds.append(f"--qcovar {covar_path}")

    run(" ".join(cmds))
    if clean_tmp:
        os.remove(pheno_path)
        if df_covar is not None:
            os.remove(covar_path)


def read_reml(path_prefix):
    """
    provide log file of the GCTA output

    read <path_prefix>.log and <path_prefix>.hsq
    """

    # Read log file
    with open(path_prefix + ".log") as f:
        lines = f.readlines()

    ## 1. find "Summary result of REML analysis:"
    line_i = (
        np.where([l.startswith("Summary result of REML analysis:") for l in lines])[
            0
        ].item()
        + 1
    )

    # read all lines before Vp
    est = []
    while 1:
        tmp = lines[line_i].strip().split("\t")
        if tmp[0] == "Vp":
            break
        est.append(tmp)
        line_i += 1
    df_est = pd.DataFrame(est[1:], columns=est[0])
    df_est = df_est.astype({"Variance": float, "SE": float})
    ## 2. find "Sampling variance/covariance of the estimates of variance components:"
    line_i = np.where(
        [
            l.startswith(
                "Sampling variance/covariance of the estimates of variance components:"
            )
            for l in lines
        ]
    )[0].item()
    varcov = [
        lines[l].strip().split("\t")
        for l in range(line_i + 1, line_i + 1 + df_est.shape[0])
    ]
    df_varcov = pd.DataFrame(
        varcov, columns=df_est["Source"].values, index=df_est["Source"].values
    )
    df_varcov = df_varcov.astype(float)

    # Read hsq file
    with open(path_prefix + ".hsq") as f:
        lines = f.readlines()
    # find loglik
    line_i = np.where([l.startswith("logL\t") for l in lines])[0].item()
    loglik = float(lines[line_i].split("\t")[1])

    # find n
    line_i = np.where([l.startswith("n\t") for l in lines])[0].item()
    n_indiv = int(lines[line_i].split("\t")[1])

    # 3. find "Log-likelihood ratio converged."
    return {"est": df_est, "varcov": df_varcov, "loglik": loglik, "n": n_indiv}