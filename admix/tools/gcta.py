import pandas as pd
import numpy as np
import subprocess
from typing import Tuple, List
import os
import glob
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
    priors: List[float] = None,
    est_fix: bool = False,
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
    if est_fix:
        cmds[0] += " --reml-est-fix"
    if priors is not None:
        cmds.append("--reml-priors " + " ".join([str(p) for p in priors]))
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


def estimate_hsq(dict_est, scale_factor=1):
    """
    Estimate the heritability, and standard error of heritability (using delta method)
    from the results of a single component REML:
    hsq = (var_g * scale_factor) / [var_g * scale_factor + var_e]

    Parameters
    ----------
    dict_est : dict
        dictionary of the results of a single component REML, as returned by read_reml
        {"est": df_est, "varcov": df_varcov}
    scale_factor : float
        scale factor for genotypical variance, default 1

    Returns
    -------
    hsq : float
        heritability
    hsq_var : float
        estimated variance of heritability
    """
    assert len(dict_est["est"]) == 2
    assert np.all(dict_est["est"].Source.values == ["V(G)", "V(e)"])
    est = dict_est["est"].Variance.values
    est_var = dict_est["varcov"].values

    ## method 1
    x, y = est[0], est[1]
    hsq = (x * scale_factor) / (x * scale_factor + y)

    # grad = [y / (x + y)^2, - x / (x + y)^2]
    grad = np.array(
        [
            scale_factor * y / ((scale_factor * x + y) ** 2),
            -scale_factor * x / ((scale_factor * x + y) ** 2),
        ]
    )

    def quad_form(x, A):
        return np.dot(np.dot(x.T, A), x)

    return hsq, quad_form(grad, est_var)


def calculate_hsq_scale(
    weight_file: str, freq_file: str, freq_col: str = "FREQ", index_col: str = "snp"
) -> float:
    """Calculate the heritability scaling factor using weight (prior variance) files
    scale_factor = sum(normalized_weight * 2 * freq * (1 - freq)), where
    normalized_weight = weight / sum(weight).

    Parameters
    ----------
    weight_file : str
        weight file
    freq_file : str
        frequency file, use a wildcard '*' to multiple frequency files
    freq_col : str
        column name of the frequency, default 'FREQ'
    index_col : str
        column name of the index, default 'snp'

    Returns
    -------
    scale_factor : float
        scaling factor

    Notes
    -----
    SNP list in weight_file should be a superset of the SNP list in freq_file.
    """
    df_weight = pd.read_csv(
        weight_file,
        sep="\t",
        index_col=0,
    )

    freq_file_list = glob.glob(freq_file)
    df_freq = (
        pd.concat(
            [
                pd.read_csv(
                    f,
                    delim_whitespace=True,
                )
                for f in freq_file_list
            ]
        ).reset_index(drop=True)
    ).set_index(index_col)

    # df_freq.index should be a superset of df_weight.index
    assert set(df_freq.index).issuperset(set(df_weight.index))
    df_weight["PRIOR_VAR"] /= df_weight["PRIOR_VAR"].sum()
    df_weight[freq_col] = df_freq[freq_col].reindex(df_weight.index)
    df_weight["FREQ_VAR"] = 2 * df_weight[freq_col] * (1 - df_weight[freq_col])

    scale_factor = np.sum(df_weight["PRIOR_VAR"] * df_weight["FREQ_VAR"])

    return scale_factor