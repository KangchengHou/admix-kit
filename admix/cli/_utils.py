import admix
import os
import subprocess
import glob
import shutil
import pandas as pd
import numpy as np
import dapgen
from typing import Tuple


def log_params(name, params):
    admix.logger.info(
        f"Received parameters: \n{name}\n  "
        + "\n  ".join(f"--{k}={v}" for k, v in params.items())
    )


def _process_sample_map(root_dir):
    """Download and format sample map from 1000 Genomes"""

    sample_map = pd.read_csv(
        f"{root_dir}/pgen/all_chr.psam",
        delim_whitespace=True,
    )
    unrelated_id = pd.read_csv(
        f"{root_dir}/pgen/king.cutoff.out.id", delim_whitespace=True
    )
    os.makedirs(f"{root_dir}/metadata", exist_ok=True)
    sample_map[["#IID", "Population", "SuperPop"]].to_csv(
        f"{root_dir}/metadata/full_sample.tsv",
        sep="\t",
        index=False,
        header=False,
    )
    # filter unrelated
    unrelated_sample_map = sample_map[~sample_map["#IID"].isin(unrelated_id["#IID"])]
    unrelated_sample_map[["#IID", "Population", "SuperPop"]].to_csv(
        f"{root_dir}/metadata/unrelated_sample.tsv",
        sep="\t",
        index=False,
        header=False,
    )
    admix.logger.info("Population in unrelated sample map:")
    admix.logger.info(unrelated_sample_map["Population"].value_counts())


def _process_genetic_map(root_dir, build):
    """
    Download and format genetic map
    1. call bash script to download genetic map to out/metadata/genetic_map/raw
    2. process the genetic map and save to out/metadata/genetic_map
    """

    assert build in ["hg19", "hg38"], "build should be hg38 or hg19"

    if build == "hg38":
        name = "GRCh38"
    elif build == "hg19":
        name = "GRCh37"
    else:
        raise ValueError("build should be hg38 or hg19")

    cmds = f"""
        mkdir -p {root_dir}/metadata/genetic_map/raw && cd {root_dir}/metadata/genetic_map/raw
        wget https://bochet.gcc.biostat.washington.edu/beagle/genetic_maps/plink.{name}.map.zip
        unzip plink.{name}.map.zip
    """

    subprocess.check_output(cmds, shell=True)

    for chrom in range(1, 23):
        # move raw files to out/metadata/genetic_map
        shutil.copy(
            f"{root_dir}/metadata/genetic_map/raw/plink.chr{chrom}.{name}.map",
            f"{root_dir}/metadata/genetic_map/chr{chrom}.map",
        )
    # clean up
    shutil.rmtree(f"{root_dir}/metadata/genetic_map/raw")

    if False:
        # OLD Eagle2 genetic map
        raw_map_path = admix.tools.get_cache_data("genetic_map", build=build)

        raw_map = pd.read_csv(
            raw_map_path,
            delim_whitespace=True,
        )

        os.makedirs(f"{root_dir}/metadata/genetic_map", exist_ok=True)
        for chrom in range(1, 23):
            chrom_map = raw_map[raw_map["chr"] == chrom].iloc[:, [0, 1, 3]]
            chrom_map.to_csv(
                f"{root_dir}/metadata/genetic_map/chr{chrom}.tsv",
                sep="\t",
                index=False,
                header=False,
            )


def get_1kg_ref(
    dir: str,
    build: str,
    verbose: bool = False,
    memory: int = 16000,
):
    """
    Get the 1,000 reference genome in plink2 format.

    Parameters
    ----------
    dir : str
        Directory where the reference genome is stored. The following files will be downloaded
        and stored in this directory:
        - ${dir}/pgen: containing plink2 files
        - ${dir}/metadata: containing metadata files
    build : str
        Build of the reference genome. hg38 or hg19.
    verbose : bool
        Whether to print out the progress of the download.
    memory : int
        Memory in MB to use for PLINK command. Default is 16000 (~16GB).

    Examples
    --------
    .. code-block:: bash

        admix get-1kg-ref --dir=/path/to/1kg --build hg38

    """
    log_params("get-1kg-ref", locals())

    def call_helper(cmd):
        admix.logger.info(f"$ {cmd}")
        if verbose:
            subprocess.run(cmd, shell=True, check=True)
        else:
            subprocess.run(
                cmd,
                shell=True,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )

    # make sure plink2 is in path
    plink2 = admix.tools.get_dependency("plink2")

    # step0: download metadata
    os.makedirs(os.path.join(dir, "metadata"), exist_ok=True)
    admix.logger.info("Retrieving meta data...")

    # step1: download pgen
    os.makedirs(os.path.join(dir, "pgen"), exist_ok=True)
    if build == "hg38":
        cmds = [
            f"mkdir -p {dir}/pgen &&",
            f"wget https://www.dropbox.com/s/j72j6uciq5zuzii/all_hg38.pgen.zst?dl=1 -O {dir}/pgen/raw.pgen.zst &&",
            f"wget https://www.dropbox.com/s/ngbo2xm5ojw9koy/all_hg38_noannot.pvar.zst?dl=1 -O {dir}/pgen/raw.pvar.zst &&",
            f"wget https://www.dropbox.com/s/2e87z6nc4qexjjm/hg38_corrected.psam?dl=1 -O {dir}/pgen/raw.psam &&",
            f"wget https://www.dropbox.com/s/4zhmxpk5oclfplp/deg2_hg38.king.cutoff.out.id?dl=1 -O {dir}/pgen/king.cutoff.out.id",
        ]
    elif build == "hg19":
        cmds = [
            f"mkdir -p {dir}/pgen &&",
            f"wget https://www.dropbox.com/s/dps1kvlq338ukz8/all_phase3_ns.pgen.zst?dl=1 -O {dir}/pgen/raw.pgen.zst &&",
            f"wget https://www.dropbox.com/s/uqk3gfhwsvf7bf3/all_phase3_ns_noannot.pvar.zst?dl=1 -O {dir}/pgen/raw.pvar.zst &&",
            f"wget https://www.dropbox.com/s/6ppo144ikdzery5/phase3_corrected.psam?dl=1 -O {dir}/pgen/raw.psam &&",
            f"wget https://www.dropbox.com/s/zj8d14vv9mp6x3c/deg2_phase3.king.cutoff.out.id?dl=1 -O {dir}/pgen/king.cutoff.out.id",
        ]
    else:
        raise ValueError(f"Unknown build: {build}")

    # assume that ${dir}/pgen already contains raw.pgen / raw.pvar / raw.psam / king.cutoff.out.id
    if not all(
        [
            os.path.exists(f"{dir}/pgen/{f}")
            for f in ["raw.pgen.zst", "raw.pvar.zst", "raw.psam"]
        ]
    ):
        admix.logger.info(
            f"Please download the reference genome to {dir}/pgen using the following commands:\n"
            + "$ "
            + " ".join(cmds)
        )
        return

    admix.logger.info("Decompressing plink files")
    cmds = [
        f"{plink2} --zst-decompress {dir}/pgen/raw.pgen.zst > {dir}/pgen/raw.pgen &&",
        f"{plink2} --zst-decompress {dir}/pgen/raw.pvar.zst > {dir}/pgen/raw.pvar",
    ]
    call_helper(" ".join(cmds))

    admix.logger.info(
        "Basic QCing: bi-allelic SNPs, MAC >= 5, chromosome 1-22, unify SNP names"
    )
    cmds = [
        f"{plink2} --pfile {dir}/pgen/raw",
        "--allow-extra-chr",
        "--rm-dup exclude-all",
        "--max-alleles 2",
        "--mac 5",
        "--snps-only",
        "--chr 1-22",
        "--set-all-var-ids @:#:\$r:\$a",
        f"--make-pgen --out {dir}/pgen/raw2",
        f"--memory {memory}",
    ]
    call_helper(" ".join(cmds))

    ## remove SNPs duplicated by position
    df_pvar = dapgen.read_pvar(f"{dir}/pgen/raw2.pvar")
    dup_snps = df_pvar.index[df_pvar.duplicated(subset=["CHROM", "POS"])]
    admix.logger.info(
        f"Excluding {len(dup_snps)}/{df_pvar.shape[0]} SNPs duplicated by positions."
    )
    np.savetxt(f"{dir}/pgen/raw2.snplist", dup_snps, fmt="%s")
    cmds = [
        f"{plink2} --pfile {dir}/pgen/raw2",
        f"--exclude {dir}/pgen/raw2.snplist",
        f"--make-pgen --out {dir}/pgen/all_chr",
        f"--memory {memory}",
    ]
    call_helper(" ".join(cmds))

    admix.logger.info("Clean up temporary files...")
    # remove raw2*
    for f in glob.glob(f"{dir}/pgen/raw2*"):
        os.remove(f)

    _process_genetic_map(root_dir=dir, build=build)
    _process_sample_map(root_dir=dir)


def select_admix_indiv(
    ref_pfile: str,
    pca_prefix: str,
    superpop1: str,
    superpop2: str,
    out: str,
    exclude_pop1: str = None,
    exclude_pop2: str = None,
    n_pc: int = 4,
    sample_t_range: Tuple[float, float] = (0.05, 0.95),
    sample_dist_max: float = 1.5,
):
    """Select admixed individuals based on joint PCA analysis results of
    (1) reference dataset (2) sample dataset

    Parameters
    ----------
    ref_pfile : str
        reference panel pfile prefix
    pca_prefix : str
        joint pca results prefix. {pca_prefix}.eigenvec, {pca_prefix}.eigenval
        will be read
    out : str
        output prefix
    superpop1 : str
        superpopulation 1
    superpop2 : str
        superpopulation 2
    exclude_pop1 : str, optional
        exclude individuals from superpopulation 1, by default None
    exclude_pop2 : str, optional
        exclude individuals from superpopulation 2, by default None
    """
    log_params("select-admixed-indiv", locals())

    df_pc, eigenval = admix.io.read_joint_pca(
        ref_pfile=ref_pfile, pca_prefix=pca_prefix
    )
    pc_cols = [f"PC{i}" for i in range(1, n_pc + 1)]
    pc_col_pos = [df_pc.columns.get_loc(col) for col in pc_cols]
    pc_eigenval = eigenval[pc_col_pos]

    df_sample_pc = df_pc[df_pc.SUPERPOP == "SAMPLE"]
    df_anc1_pc = df_pc[df_pc.SUPERPOP == superpop1]
    df_anc2_pc = df_pc[df_pc.SUPERPOP == superpop2]

    if exclude_pop1 is not None:
        df_anc1_pc = df_anc1_pc[~df_anc1_pc.POP.isin(exclude_pop1)]
    if exclude_pop2 is not None:
        df_anc2_pc = df_anc2_pc[~df_anc2_pc.POP.isin(exclude_pop2)]

    df_sample_pc, df_anc1_pc, df_anc2_pc = (
        df_sample_pc[pc_cols],
        df_anc1_pc[pc_cols],
        df_anc2_pc[pc_cols],
    )

    # prepare sample_pc, anc1_pc, anc2_pc
    sample_dist, sample_t = admix.data.distance_to_refpop(
        df_sample_pc, df_anc1_pc, df_anc2_pc, weight=np.sqrt(pc_eigenval)
    )

    selected_mask = (
        (sample_t_range[0] < sample_t)
        & (sample_t < sample_t_range[1])
        & (sample_dist < sample_dist_max)
    )
    selected_indiv = df_sample_pc.index[selected_mask]
    admix.logger.info(
        f"{len(selected_indiv)}/{len(df_sample_pc)} selected to be admixed individuals."
    )

    df_plot = pd.concat([df_pc[df_pc.SUPERPOP != "SAMPLE"], df_pc.loc[selected_indiv]])
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(figsize=(8.5, 4), dpi=150, ncols=2)
    admix.plot.joint_pca(df_pc=df_plot, eigenval=eigenval, axes=axes)
    fig.tight_layout()
    fig.savefig(f"{out}.png", bbox_inches="tight")
    np.savetxt(f"{out}.indiv", selected_indiv, fmt="%s")
    admix.logger.info(f"PCA plots saved to {out}.png.")
    admix.logger.info(f"Selected individuals saved to {out}.indiv.")
