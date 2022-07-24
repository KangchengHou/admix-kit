import admix
import os
from admix.utils import cd
import subprocess
import glob
import shutil
import pandas as pd


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
        raw_map = pd.read_csv(
            f"{root_dir}/metadata/genetic_map/raw/plink.chr{chrom}.{name}.map",
            delim_whitespace=True,
            header=None,
        )
        raw_map = raw_map[[0, 3, 2]]
        raw_map.to_csv(
            f"{root_dir}/metadata/genetic_map/{build}.chr{chrom}.tsv",
            sep="\t",
            index=False,
            header=False,
        )
    # clean up
    shutil.rmtree(f"{root_dir}/metadata/genetic_map/raw")


def get_1kg_ref(dir: str, build: str = "hg38", verbose: bool = False, step: int = None):
    """
    Get the 1,000 reference genome in plink2 format.

    Parameters
    ----------
    dir : str
        Directory where the reference genome is stored. The following files will be
        downloaded and stored in this directory:
        - ${dir}/pgen: containing plink2 files
        - ${dir}/vcf: containing vcf files
        - ${dir}/metadata: containing metadata files
    build : str
        Build of the reference genome. hg38 (default) or hg19.
    verbose : bool
        Whether to print out the progress of the download.
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

    # make sure plink2 / tabix is in path
    assert shutil.which("plink2") is not None, "plink2 is not in $PATH"
    assert shutil.which("tabix") is not None, "tabix is not in $PATH"

    # assert dir not exist
    assert not os.path.exists(dir), f"dir='{dir}' already exists"

    # step1: download pgen
    os.makedirs(os.path.join(dir, "pgen"))
    if build == "hg38":
        cmds = [
            f"wget https://www.dropbox.com/s/23xlpscis1p5xud/all_hg38_ns.pgen.zst?dl=1 -O {dir}/pgen/raw.pgen.zst &&",
            f"wget https://www.dropbox.com/s/hy54ba9yvw665xf/all_hg38_ns_noannot.pvar.zst?dl=1 -O {dir}/pgen/raw.pvar.zst &&",
            f"wget https://www.dropbox.com/s/3j9zg103fi8cjfs/hg38_corrected.psam?dl=1 -O {dir}/pgen/raw.psam &&",
            f"wget https://www.dropbox.com/s/129gx0gl2v7ndg6/deg2_hg38.king.cutoff.out.id?dl=1 -O {dir}/pgen/king.cutoff.out.id",
        ]
    elif build == "hg19":
        cmds = [
            f"wget https://www.dropbox.com/s/dps1kvlq338ukz8/all_phase3_ns.pgen.zst?dl=1 -O {dir}/pgen/raw.pgen.zst &&",
            f"wget https://www.dropbox.com/s/uqk3gfhwsvf7bf3/all_phase3_ns_noannot.pvar.zst?dl=1 -O {dir}/pgen/raw.pvar.zst &&",
            f"wget https://www.dropbox.com/s/6ppo144ikdzery5/phase3_corrected.psam?dl=1 -O {dir}/pgen/raw.psam &&",
            f"wget https://www.dropbox.com/s/zj8d14vv9mp6x3c/deg2_phase3.king.cutoff.out.id?dl=1 -O {dir}/pgen/king.cutoff.out.id",
        ]
    else:
        raise ValueError(f"Unknown build: {build}")
    admix.logger.info("Downloading plink2 files...")
    call_helper(" ".join(cmds))

    # decompress pgen
    admix.logger.info("Decompressing plink2 files...")
    cmds = [
        f"plink2 --zst-decompress {dir}/pgen/raw.pgen.zst > {dir}/pgen/raw.pgen &&",
        f"plink2 --zst-decompress {dir}/pgen/raw.pvar.zst > {dir}/pgen/raw.pvar",
    ]
    call_helper(" ".join(cmds))

    admix.logger.info(
        "Basic QCing: bi-allelic SNPs, MAC >= 5, chromosome 1-22, unify SNP names"
    )
    cmds = [
        f"plink2 --pfile {dir}/pgen/raw",
        "--allow-extra-chr",
        "--rm-dup exclude-all",
        "--max-alleles 2",
        "--mac 5",
        "--snps-only",
        "--chr 1-22",
        "--set-all-var-ids @:#:\$r:\$a",
        f"--make-pgen --out {dir}/pgen/all_chr",
    ]
    call_helper(" ".join(cmds))

    admix.logger.info("Clean up temporary files...")
    # remove raw*
    for f in glob.glob(f"{dir}/pgen/raw*"):
        os.remove(f)

    # step2: convert plink2 to vcf
    os.makedirs(os.path.join(dir, "vcf"))
    admix.logger.info("Converting plink2 to vcf...")

    for chrom in range(1, 23):
        cmds = [
            f"plink2 --pfile {dir}/pgen/all_chr",
            "--export vcf bgz",
            f"--chr {chrom}",
            f"--out {dir}/vcf/chr{chrom} && tabix -p vcf {dir}/vcf/chr{chrom}.vcf.gz",
        ]
        call_helper(" ".join(cmds))

    # step3: download metadata
    os.makedirs(os.path.join(dir, "metadata"))
    admix.logger.info("Downloading meta data...")

    _process_sample_map(root_dir=dir)
    _process_genetic_map(root_dir=dir, build=build)