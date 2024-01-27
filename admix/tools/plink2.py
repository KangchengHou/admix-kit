import dapgen
import pandas as pd
import admix
import numpy as np

import dapgen
import pandas as pd
import admix
import numpy as np
from typing import List
import os
import glob
import subprocess
from ._utils import get_dependency


def run(cmd: str, **kwargs):
    """Shortcut for running plink commands

    Parameters
    ----------
    cmd : str
        plink command
    """
    bin_path = get_dependency("plink2")
    add_cmds = [f" --{k.replace('_', '-')} {kwargs[k]}" for k in kwargs]
    cmd += " ".join(add_cmds)
    subprocess.check_call(f"{bin_path} {cmd}", shell=True)


def gwas(
    df_sample_info: pd.DataFrame,
    pheno_col: str,
    out_prefix: str,
    pfile: str = None,
    bfile: str = None,
    family: str = "linear",
    covar_cols: List[str] = None,
    cat_cols: List[str] = None,
    pheno_quantile_normalize=False,
    covar_quantile_normalize=False,
    clean_tmp_file=False,
    **kwargs,
):
    assert family in ["linear", "logistic"], "family must be linear or logistic"
    # only one of pfile or bfile must be provided
    assert (pfile is None) != (bfile is None), "pfile or bfile must be provided"

    # clean up phenotype
    non_nan_index = ~np.isnan(df_sample_info[pheno_col])
    df_pheno = df_sample_info.loc[non_nan_index, :].rename(columns={pheno_col: "trait"})
    pheno_vals = df_pheno.trait.values
    if family == "logistic":
        assert (
            pheno_quantile_normalize == False
        ), "quantile normalization can not be used for logistic regression"
        assert np.all(
            (pheno_vals == 0) | (pheno_vals == 1)
        ), "phenotype values must be 0 or 1 for logistic regression"
        df_pheno["trait"] = df_pheno.trait.astype(int) + 1
    elif family == "linear":
        df_pheno["trait"] = (pheno_vals - pheno_vals.mean()) / pheno_vals.std()
    else:
        raise ValueError("family must be linear or logistic")
    pheno_path = out_prefix + f".plink2_tmp_pheno"

    if bfile is not None:
        # PLINK1 format
        # FID, IID must be in the column
        assert "FID" in df_sample_info.columns
        assert "IID" in df_sample_info.columns
        df_pheno[["FID", "IID", "trait"]].to_csv(
            pheno_path, sep="\t", index=False, na_rep="NA"
        )
        cmds = [f"--bfile {bfile}"]
    elif pfile is not None:
        # PLINK2 format
        df_pheno.index.name = "#IID"
        df_pheno[["trait"]].to_csv(pheno_path, sep="\t", na_rep="NA")
        cmds = [f"--pfile {pfile}"]

    cmds.extend(
        [
            f"--pheno {pheno_path} --no-psam-pheno",
            f"--out {out_prefix}",
        ]
    )
    if family == "linear":
        cmds.append("--linear hide-covar omit-ref")
    elif family == "logistic":
        cmds.append("--logistic hide-covar omit-ref no-firth")
    else:
        raise ValueError("family must be linear or logistic")

    if covar_cols is not None:
        covar_path = out_prefix + f".plink2_tmp_covar"
        cmds.append(f"--covar {covar_path}")
        if bfile is not None:
            # PLINK1 format
            # FID, IID must be in the column
            assert "FID" in df_sample_info.columns
            assert "IID" in df_sample_info.columns
            df_covar = df_sample_info.loc[
                non_nan_index, ["FID", "IID"] + covar_cols
            ].copy()
            df_covar.to_csv(covar_path, sep="\t", index=False, na_rep="NA")
        elif pfile is not None:
            # PLINK2 format
            df_covar = df_sample_info.loc[non_nan_index, covar_cols].copy()
            df_covar.index.name = "#IID"
            df_covar.to_csv(covar_path, sep="\t", na_rep="NA")

    else:
        cmds[-1] += " allow-no-covars"

    if pheno_quantile_normalize:
        cmds.append("--pheno-quantile-normalize")
    if covar_quantile_normalize:
        cmds.append("--covar-quantile-normalize")
    if cat_cols:
        cmds.append(f"--split-cat-pheno omit-most {' '.join(cat_cols)}")

    print("\n".join(cmds))

    run(" ".join(cmds), **kwargs)

    if family == "linear":
        os.rename(out_prefix + ".trait.glm.linear", out_prefix + ".assoc")
    else:
        os.rename(out_prefix + ".trait.glm.logistic", out_prefix + ".assoc")

    if clean_tmp_file:
        for f in glob.glob(out_prefix + ".plink2_tmp_*"):
            os.remove(f)


def clump(
    pfile,
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
    For now, first need to export to .bed format then perform the clump
    """
    tmp_prefix = out_prefix + ".admix_plink2_clump_tmp"
    # convert to bed
    admix.tools.plink2.run(f"--pfile {pfile} --make-bed --out {tmp_prefix}")

    # perform clump
    admix.tools.plink.clump(
        bfile=tmp_prefix,
        assoc_path=assoc_path,
        out_prefix=out_prefix,
        p1=p1,
        p2=p2,
        r2=r2,
        kb=kb,
        **kwargs,
    )
    # # convert plink2 association to plink1 format ID -> SNP
    # import shutil

    # from_file = open(assoc_path)
    # to_file = open(tmp_prefix + ".assoc", "w")
    # to_file.writelines(from_file.readline().replace("ID", "SNP"))
    # shutil.copyfileobj(from_file, to_file)
    # from_file.close()
    # to_file.close()
    # cmds = [
    #     f"--bfile {tmp_prefix} --clump {tmp_prefix + '.assoc'}",
    #     f"--clump-p1 {p1} --clump-p2 {p2} --clump-r2 {r2} --clump-kb {kb}",
    #     f"--out {tmp_prefix}",
    # ]

    # admix.tools.plink.run(" ".join(cmds))
    # if os.path.exists(tmp_prefix + ".clumped"):
    #     os.rename(tmp_prefix + ".clumped", out_prefix + ".clumped")
    # else:
    #     # no clumped region
    #     # write a comment to the output file
    #     with open(out_prefix + ".clumped", "w") as file:
    #         file.write("# No clumped region")

    for f in glob.glob(tmp_prefix + "*"):
        os.remove(f)


def lift_over(pfile: str, out_prefix: str, chain="hg19->hg38"):
    """Lift over a plink file to another genome. First, use liftover to convert to
    another genome, then use plink2 --sort-vars to sort the variants.

    Parameters
    ----------
    pfile : str
        Path to plink file
    out_prefix : str
        output prefix, <out_prefix>.pgen, <out_prefix>.psam, <out_prefix>.pvar will
        be generated
    chain : str, optional
        lift over direction, by default "hg19->hg38"
    """
    assert chain in ["hg19->hg38", "hg38->hg19"]

    # read the input pgen
    pgen, pvar, psam = dapgen.read_pfile(pfile)
    df_snp = pd.DataFrame(
        {"CHROM": pvar.CHROM.values, "POS": pvar.POS.values}, index=pvar.index.values
    )
    # perform the liftover
    df_lifted = admix.tools.liftover.run(df_snp, chain=chain)
    n_snp1 = len(df_lifted)
    df_lifted = df_lifted[df_lifted.POS != -1]
    n_snp2 = len(df_lifted)

    print(
        f"remove {n_snp1 - n_snp2} / {n_snp1} SNPs"
        f" ({(n_snp1 - n_snp2) / n_snp1 * 100:.2g}%)"
        " for unmapped or ambiguous SNPs"
    )

    # extract the lifted SNPs
    np.savetxt(f"{out_prefix}-tmp.snp", df_lifted.index.values, fmt="%s")
    admix.tools.plink2.run(
        f"--pfile {pfile} --extract {out_prefix}-tmp.snp --sort-vars --make-pgen --out {out_prefix}-tmp"
    )

    # substitute the coordinates obtained from liftover
    pgen, pvar, psam = dapgen.read_pfile(out_prefix + "-tmp")
    assert np.all(pvar.index == df_lifted.index)

    pvar["POS"] = df_lifted["POS"]
    pvar.insert(2, "ID", pvar.index)
    pvar = pvar.reset_index(drop=True).rename(columns={"CHROM": "#CHROM"})
    pvar.to_csv(f"{out_prefix}-tmp.pvar", sep="\t", index=False)

    admix.tools.plink2.run(
        f"--pfile {out_prefix}-tmp --sort-vars --make-pgen --out {out_prefix}"
    )
    # remove the tmp files
    for f in glob.glob(out_prefix + "-tmp*"):
        os.remove(f)


def align_snp(pfile1: str, pfile2: str, out_prefix: str):
    """Given two plink files from different individuals, take the common set of SNPs.
    Matching by (1) chromosome (2) position (3) ref allele (4) alt allele

    This can be useful for software that requires an aligned set of SNPs

    Parameters
    ----------
    pfile1 : str
        plink file 1
    pfile2 : str
        plink file 2
    out_prefix: str
        prefix to the output files
    """
    # Step 1: read pvar and find the common SNPs
    df_snp1 = dapgen.read_pvar(pfile1 + ".pvar").drop_duplicates(
        subset=["CHROM", "POS"]
    )
    df_snp2 = dapgen.read_pvar(pfile2 + ".pvar").drop_duplicates(
        subset=["CHROM", "POS"]
    )

    idx1, idx2, flip = dapgen.align_snp(df_snp1, df_snp2)

    admix.logger.info(
        f"{len(idx1)} / {len(df_snp1)} ({len(idx1) / len(df_snp1) * 100:.3g}%) SNPs extracted in pfile1: {pfile1}"
    )
    admix.logger.info(
        f"{len(idx2)} / {len(df_snp2)} ({len(idx2) / len(df_snp2) * 100:.3g}%) SNPs extracted in pfile2: {pfile2}"
    )

    # Step 2: extract the common SNPs from the two files and fix the allele order
    allele1 = df_snp1.loc[idx1]["REF"]
    allele2 = df_snp2.loc[idx2]["REF"]

    allele1.to_csv(out_prefix + "-tmp.1.refallele", sep="\t", header=False)
    np.savetxt(out_prefix + "-tmp.1.snplist", allele1.index.values, fmt="%s")

    # use sample ref-allele values, but use ref-allele index
    pd.Series(allele1.values, index=allele2.index).to_csv(
        out_prefix + "-tmp.2.refallele", sep="\t", header=False
    )
    np.savetxt(out_prefix + "-tmp.2.snplist", allele2.index.values, fmt="%s")

    # extract common SNPs and align REF ALT
    admix.tools.plink2.run(
        f"--pfile {pfile1} --extract {out_prefix + '-tmp.1.snplist'}"
        f" --ref-allele {out_prefix + '-tmp.1.refallele'} 2 1 --make-pgen --out {out_prefix}.1"
    )
    admix.tools.plink2.run(
        f"--pfile {pfile2} --extract {out_prefix + '-tmp.2.snplist'} "
        f" --ref-allele force {out_prefix + '-tmp.2.refallele'} 2 1 --make-pgen --out {out_prefix}.2"
    )

    # remove the tmp files
    for f in glob.glob(out_prefix + "-tmp*"):
        os.remove(f)


def merge_indiv(pfile1: str, pfile2: str, out_prefix: str):
    """Given two plink files from different individuals, take the common set of SNPs
    and merge them into one data set.

    This is a wrapper for various PLINK functions. Since PLINK2's pmerge function is
    not complete now. We use PLINK1 merge functionality instead. A PLINK1 bfile is
    produced (rather than PLINK2 pfile).

    This can be useful for example when we want to perform a joint PCA for the two
    data sets.

    Parameters
    ----------
    pfile1 : str
        plink file 1
    pfile2 : str
        plink file 2
    out_prefix: str
        prefix to the output files
    """

    admix.tools.plink2.align_snp(
        pfile1=pfile1, pfile2=pfile2, out_prefix=out_prefix + "-tmp"
    )

    # Step 3: unify SNP ID
    admix.tools.plink2.run(
        f"--pfile {out_prefix}-tmp.1 --set-all-var-ids @:#:\$r:\$a --make-bed --out {out_prefix}-tmp.1.fixid"
    )
    admix.tools.plink2.run(
        f"--pfile {out_prefix}-tmp.2 --set-all-var-ids @:#:\$r:\$a --make-bed --out {out_prefix}-tmp.2.fixid"
    )

    # Step 4: make final merged bfile
    admix.tools.plink.run(
        f"--bfile {out_prefix}-tmp.1.fixid --bmerge {out_prefix}-tmp.2.fixid --make-bed --out {out_prefix}"
    )

    # remove the tmp files
    for f in glob.glob(out_prefix + "-tmp*"):
        os.remove(f)


def subset(
    pfile: str,
    out_prefix: str,
    snp_list: List = None,
    indiv_list: List = None,
):
    cmds = [f"--pfile {pfile}", f"--make-pgen --out {out_prefix}", "--memory 16000"]

    if snp_list is not None:
        snplist_path = out_prefix + ".admix-plink2-subset.snplist"
        np.savetxt(snplist_path, snp_list, fmt="%s")

        cmds.append(f"--extract {snplist_path}")

    if indiv_list is not None:
        indivlist_path = out_prefix + ".admix-plink2-subset.indivlist"
        np.savetxt(indivlist_path, indiv_list, fmt="%s")
        cmds.append(f"--keep {indivlist_path}")

    run(" ".join(cmds))

    # clean up
    for f in glob.glob(out_prefix + ".admix-plink2-subset.*"):
        os.remove(f)


# TODO: check difference with plink2.gwas, consider merging these two functions
def assoc(
    bfile: str,
    pheno: pd.DataFrame,
    out_prefix: str,
    covar: pd.DataFrame = None,
    indiv: pd.DataFrame = None,
    snp: pd.DataFrame = None,
):
    """Run plink association test

    Parameters
    ----------
    bfile : str
        plink binary file
    pheno : pd.DataFrame
        phenotype data
    out_prefix : str
        prefix for output files
    covar : pd.DataFrame, optional
        covariate data
    indiv : pd.DataFrame, optional
        individual to subset, should contain columns FID and IID
    snp : pd.DataFrame, optional
        snp to subset, should contain column SNP
    """

    assert indiv is None and snp is None, "indiv and snp are not supported"
    pheno_path = out_prefix + ".tmp_pheno"
    covar_path = out_prefix + ".tmp_covar"
    pheno.to_csv(pheno_path, sep="\t", index=False)
    if covar is not None:
        covar.to_csv(covar_path, sep="\t", index=False)

    cmd = [
        f"--bfile {bfile}",
        f"--out {out_prefix}",
        f"--pheno {pheno_path}",
    ]
    if covar is not None:
        cmd.append("--ci 0.95 --glm omit-ref hide-covar")
        cmd.append(f"--covar {covar_path}")
    else:
        cmd.append("--ci 0.95 --glm omit-ref hide-covar allow-no-covars")

    run(" ".join(cmd))


def prune(
    pfile: str,
    out_prefix: str,
    indep_params: List = None,
    indep_pairwise_params: List = None,
):
    """Run plink2 prune

    For example, indep_params = [200, 5, 1.15], indep-pairwise = [100 5 0.1], then two-step
    pruning will be performed.

    Parameters
    ----------
    pfile : str
        plink binary file
    indep_params : List
        list of parameters for indep
    indep_pairwise_params : List
        list of parameters for indep-pairwise
    out_prefix : str
        prefix for output files
    """

    assert indep_params is None, "indep_params is not supported yet"

    tmp_prefix = out_prefix + ".admix_plink2_prune_tmp"

    assert (indep_params is None) != (
        indep_pairwise_params is None
    ), "only one of indep_params and indep_pairwise_params can be specified"

    # step 1: indep pruning
    cmd = [
        f"--pfile {pfile}",
        f"--out {tmp_prefix}",
    ]

    if indep_params is not None:
        cmd.append(f"--indep {' '.join([str(p) for p in indep_params])}")
    if indep_pairwise_params is not None:
        cmd.append(
            f"--indep-pairwise {' '.join([str(p) for p in indep_pairwise_params])}"
        )

    run(" ".join(cmd))

    cmd = [
        f"--pfile {pfile}",
        f"--extract {tmp_prefix}.prune.in",
        f"--make-pgen --out {out_prefix}",
    ]
    run(" ".join(cmd))

    # clean up
    for f in glob.glob(tmp_prefix + ".*"):
        os.remove(f)


def pca(pfile: str, out_prefix: str, approx: bool = False, **kwargs):
    """Run plink2 pca

    TODO: include more options

    Parameters
    ----------
    pfile : str
        plink binary file
    out_prefix : str
        prefix for output files
    approx: bool
    """

    cmd = [
        f"--pfile {pfile}",
        f"--out {out_prefix}",
    ]
    if approx:
        cmd.append("--pca approx")
    else:
        cmd.append("--pca")
    run(" ".join(cmd), **kwargs)


def subset_hapmap3(pfile: str, build: str, out_prefix: str = None, chrom: int = None):
    """Overlap a pfile with HapMap3 SNPs

    Parameters
    ----------
    pfile : str
        input pfile
    build: str
        hg19 or hg38
    out_prefix: str
        output pfile prefix

    Returns
    -------
    <out_prefix>.{pgen|pvar|psam} will be written to disk if out_prefix is specified
    otherwise, snp_list will be returned

    """
    assert build in ["hg19", "hg38"], "build must be hg19 or hg38"
    try:
        import pyreadr
    except ImportError:
        raise ImportError("pyreadr is not installed. Use pip install pyreadr")
    hm3_snps = pyreadr.read_r(admix.tools.get_cache_data("hapmap3_snps"))[None]
    if build == "hg19":
        hm3_snps = hm3_snps[["chr", "pos"]].rename(
            columns={"chr": "CHROM", "pos": "POS"}
        )
    elif build == "hg38":
        hm3_snps = hm3_snps[["chr", "pos_hg38"]].rename(
            columns={"chr": "CHROM", "pos_hg38": "POS"}
        )
    else:
        raise NotImplementedError

    TOTAL_HM3_SNPS = hm3_snps.shape[0]
    if chrom is not None:
        hm3_snps = hm3_snps[hm3_snps.CHROM == chrom]

    df_snp = dapgen.read_pvar(pfile + ".pvar")
    snp_list = df_snp.reset_index().merge(hm3_snps, on=["CHROM", "POS"])["snp"].values

    admix.logger.info(f"{len(snp_list)}/{TOTAL_HM3_SNPS} SNPs are retained")
    if out_prefix is None:
        return snp_list
    else:
        subset(pfile, out_prefix, snp_list=snp_list)
