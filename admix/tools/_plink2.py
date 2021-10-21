import xrpgen
import pandas as pd
import admix
import numpy as np
from typing import List
import os
import glob


def plink2_gwas(
    pfile,
    df_sample_info: pd.DataFrame,
    pheno_col: str,
    out_prefix: str,
    covar_cols: List[str] = None,
    cat_cols: List[str] = None,
    pheno_quantile_normalize=False,
    covar_quantile_normalize=False,
    clean_tmp_file=False,
):

    non_nan_index = ~np.isnan(df_sample_info[pheno_col])
    df_pheno = (
        df_sample_info.loc[non_nan_index, [pheno_col]]
        .copy()
        .rename(columns={pheno_col: "trait"})
    )
    df_pheno.index.name = "#IID"

    pheno_path = out_prefix + f".plink2_tmp_pheno"
    df_pheno.to_csv(pheno_path, sep="\t", na_rep="NA")

    cmds = [
        f"--pfile {pfile}",
        f"--pheno {pheno_path}",
        "--linear hide-covar",
        f"--out {out_prefix}",
    ]

    if covar_cols is not None:
        df_covar = df_sample_info.loc[non_nan_index, covar_cols].copy()
        df_covar.index.name = "#IID"

        covar_path = out_prefix + f".plink2_tmp_covar"
        df_covar.to_csv(covar_path, sep="\t", na_rep="NA")
        cmds.append(f"--covar {covar_path}")
    else:
        cmds[2] += " allow-no-covars"

    if pheno_quantile_normalize:
        cmds.append("--pheno-quantile-normalize")
    if covar_quantile_normalize:
        cmds.append("--covar-quantile-normalize")
    if cat_cols:
        cmds.append(f"--split-cat-pheno omit-most {' '.join(cat_cols)}")

    print("\n".join(cmds))

    admix.tools.plink2(" ".join(cmds))
    os.rename(out_prefix + ".trait.glm.linear", out_prefix + ".assoc")
    if clean_tmp_file:
        for f in glob.glob(out_prefix + ".plink2_tmp_*"):
            os.remove(f)


def plink2_clump(pfile, assoc_path: str, out_prefix: str, p1: float = 5e-8):
    """
    Wrapper for plink2 clump
    For now, first need to export to .bed format then perform the clump
    """
    tmp_prefix = out_prefix + ".plink2_tmp"
    # convert to bed
    admix.tools.plink2(f"--pfile {pfile} --make-bed --out {tmp_prefix}")

    # convert plink2 association to plink1 format ID -> SNP
    import shutil

    from_file = open(assoc_path)
    to_file = open(tmp_prefix + ".assoc", "w")
    to_file.writelines(from_file.readline().replace("ID", "SNP"))
    shutil.copyfileobj(from_file, to_file)
    from_file.close()
    to_file.close()

    admix.tools.plink(
        f"--bfile {tmp_prefix} --clump {tmp_prefix + '.assoc'} --clump-p1 {p1} --out {tmp_prefix}"
    )
    os.rename(tmp_prefix + ".clumped", out_prefix + ".clumped")

    for f in glob.glob(tmp_prefix + "*"):
        os.remove(f)


def plink2_lift_over(pfile: str, out_prefix: str, chain="hg19->hg38"):
    assert chain in ["hg19->hg38", "hg38->hg19"]

    # read the input pgen
    pgen, pvar, psam = xrpgen.read_pfile(pfile)
    df_snp = pd.DataFrame(
        {"CHROM": pvar.CHROM.values, "POS": pvar.POS.values}, index=pvar.index.values
    )
    # perform the liftover
    df_lifted = admix.tools.lift_over(df_snp, chain=chain)
    n_snp1 = len(df_lifted)
    df_lifted = df_lifted[df_lifted.POS != -1]
    n_snp2 = len(df_lifted)

    print(
        f"remove {n_snp1 - n_snp2} SNPs ({(n_snp1 - n_snp2) / n_snp1 * 100:.2g}%) for unmapped or ambiguous SNPs"
    )

    # extract the lifted SNPs
    np.savetxt(f"{out_prefix}.snp", df_lifted.index.values, fmt="%s")
    admix.tools.plink2(
        f"--pfile {pfile} --extract {out_prefix}.snp --sort-vars --make-pgen --out {out_prefix}"
    )

    # substitute the coordinates
    pgen, pvar, psam = xrpgen.read_pfile(out_prefix)
    assert np.all(pvar.index == df_lifted.index)

    pvar["POS"] = df_lifted["POS"]
    pvar.insert(2, "ID", pvar.index)
    pvar = pvar.reset_index(drop=True).rename(columns={"CHROM": "#CHROM"})
    pvar.to_csv(f"{out_prefix}.pvar", sep="\t", index=False)


def plink2_merge(sample_pfile: str, ref_pfile: str, out_prefix: str):
    """Given two plink files from different samples, take the common set of SNPs
    and merge them into one data set.

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

    # TODO: cope with allele flop in the sample pfile

    # Step 1: rename the SNPstwo files
    admix.tools.plink2(
        f"--pfile {sample_pfile} --set-all-var-ids @:#:\$r:\$a --make-pgen --out {out_prefix}"
    )

    admix.tools.plink2(
        f"--pfile {ref_pfile} --set-all-var-ids @:#:\$r:\$a --make-pgen --out {out_prefix}"
    )

    # Step 2: find common SNPs
    # load in the SNPs and find the intersection

    # Step 3: extract SNPs and merge the data
