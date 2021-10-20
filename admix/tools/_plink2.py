import xrpgen
import pandas as pd
import admix
import numpy as np


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
