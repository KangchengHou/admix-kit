"""
Standalone functions corresponding to the external tools
"""

import admix
import tempfile
import os
import subprocess
import pandas as pd
import shutil
from typing import List
import numpy as np
import dapgen
from . import get_dependency, get_cache_data


def hapgen2(
    pfile: str,
    chrom: int,
    n_indiv: int,
    out_prefix: str,
    genetic_map: str = "hg38",
    plink_kwargs: dict = dict(),
):
    """simulate genotype with HAPGEN2

    Parameters
    ----------
    pfile : str
        path to PLINK2 pfile
    chrom : int
        chromosome number, even if pfile contains 1 chromosome, it still need to be
        specified
    genetic_map : str
        genetic map, downloaded from
        https://alkesgroup.broadinstitute.org/Eagle/downloads/tables/
    n_indiv : int
        Number of individuals to simulate
    plink_kwargs : dict
        kwargs to pass to plink2, e.g. --from-bp XX --to-bp YY
    out_prefix : str
        output prefix
    """
    tmp = tempfile.TemporaryDirectory()
    tmp_dir = tmp.name
    tmp_data_prefix = os.path.join(tmp_dir, "hapgen2_data")

    ##################################
    # convert PLINK2 to legend and hap
    ##################################

    admix.tools.plink2.run(
        " ".join(
            [
                "--pfile",
                pfile,
                "--export hapslegend",
                f"--out {tmp_data_prefix}",
                f"--chr {chrom}",
            ]
        ),
        **plink_kwargs,
    )

    ##################################
    # simulate genotype with HAPGEN2
    ##################################

    assert genetic_map in ["hg19", "hg38"]
    df_map = pd.read_csv(
        get_cache_data("genetic_map", build=genetic_map), delim_whitespace=True
    )
    df_map = df_map[df_map.chr == chrom].drop(columns=["chr"])
    df_map.to_csv(f"{tmp_data_prefix}.genetic_map", sep="\t", index=False)

    df_legend = pd.read_csv(f"{tmp_data_prefix}.legend", delim_whitespace=True, nrows=1)
    dl = df_legend["position"][0]

    hapgen2 = get_dependency("hapgen2")

    cmd_hapgen2 = [
        f"{hapgen2}",
        f" -m {tmp_data_prefix}.genetic_map",
        f" -l {tmp_data_prefix}.legend",
        f" -h {tmp_data_prefix}.haps",
        f" -o {tmp_data_prefix}.hapgen2.gz",
        f" -dl {dl} 1 1 1",
        f" -n {n_indiv} 0",
        " -no_gens_output",
    ]
    subprocess.check_call(" ".join(cmd_hapgen2), shell=True)

    ##################################
    # convert HAPGEN2 to PLINK2
    ##################################

    cmd_convert_pgen = [
        f"--haps {tmp_data_prefix}.hapgen2.controls.haps.gz",
        f"--legend {tmp_data_prefix}.hapgen2.legend {chrom}",
        "--make-pgen",
        f"--out {out_prefix}",
    ]

    admix.tools.plink2.run(" ".join(cmd_convert_pgen))

    shutil.move(f"{tmp_data_prefix}.hapgen2.gz.summary", f"{out_prefix}.hapgen2.log")

    ##################################
    # clean up
    ##################################
    tmp.cleanup()


def interpolate_genetic_position(chrom: int, pos: np.ndarray, build: str) -> np.ndarray:
    """Intepolate physical position (bp) to genetic position (cM)

    Parameters
    ----------
    chrom : int
        which chromosome is the position `pos` on. Pass SNP position one chromosome
        at a time.
    pos : np.ndarray
        position
    build : str
        genome build

    Returns
    -------
    np.ndarray
        genetic position
    """
    assert build in ["hg19", "hg38"]
    from scipy.interpolate import interp1d

    df_map = pd.read_csv(
        admix.tools.get_cache_data("genetic_map", build=build), delim_whitespace=True
    )
    df_map = df_map[df_map["chr"] == chrom]
    assert np.all(np.sort(df_map["position"]) == df_map["position"])
    interp = interp1d(
        x=df_map["position"], y=df_map["Genetic_Map(cM)"], assume_sorted=True
    )
    genetic_pos = interp(pos)
    return genetic_pos


def admix_simu(
    pfile_list: List[str],
    admix_prop: List[float],
    n_gen: int,
    n_indiv: int,
    build: str,
    out_prefix: str,
    admix_simu_dir: str = None,
):
    """
    Wrapper for https://github.com/williamslab/admix-simu

    Parameters
    ----------
    pfile_list : List[str]
        list of input prefixes
    admix_prop :
        proportion of admixed individuals relative proportion of the admixture,
        will be rescaled to 1.
    n_gen : int
        number of generations to simulate
    n_indiv : int
        number of individuals to simulate
    out_prefix: output prefix
        admix_simu_dir: Directory to the admix_simu software

    Returns
    -------

    haps format
    -----------
    https://mathgen.stats.ox.ac.uk/genetics_software/shapeit/shapeit.html#hapsample

    phgeno format
    -------------
    <i1_hap1><i1_hap2><i2_hap1><i2_hap2>

    Steps
    -----
    1. Convert to phgeno format
    2. Run admix-simu
    3. Interpolate genetic position https://privefl.github.io/bigsnpr/reference/snp_asGeneticPos.html

    """

    ##################################
    # check input
    ##################################
    assert build in ["hg19", "hg38"]
    assert len(pfile_list) == len(
        admix_prop
    ), "`pfile_list`, `admix_prop` should have the same length"
    n_pop = len(pfile_list)

    # TODO: support arbitrary number of populations
    assert n_pop == 2, "Currently we only support two-way admixture"

    ##################################
    # format input
    ##################################
    tmp = tempfile.TemporaryDirectory()
    tmp_dir = tmp.name
    df_snp_info = None
    chrom = None
    for pfile_path in pfile_list:
        pfile = os.path.basename(pfile_path)
        print(pfile)
        admix.tools.plink2.run(
            f"--pfile {pfile_path} --export hapslegend --out {tmp_dir}/{pfile}"
        )
        # convert haps to phgeno by removing all spaces
        subprocess.check_output(
            f"cat {tmp_dir}/{pfile}.haps | tr -d ' ' > {tmp_dir}/{pfile}.phgeno",
            shell=True,
        )
        # check all legend files are the same
        if df_snp_info is None:
            df_snp_info = pd.read_csv(
                f"{tmp_dir}/{pfile}.legend", delim_whitespace=True
            )
        else:
            assert df_snp_info.equals(
                pd.read_csv(f"{tmp_dir}/{pfile}.legend", delim_whitespace=True)
            ), f"SNP information in {pfile} are not the same as in {pfile_list[0]}"

        # check all chromosomes are the same and only one chromosome
        tmp_chrom = dapgen.read_pvar(f"{pfile_path}.pvar")["CHROM"]
        assert len(set(tmp_chrom)) == 1, f"Multiple chromosomes in {pfile_path}.pvar"
        if chrom is None:
            chrom = tmp_chrom[0]
        else:
            assert (
                chrom == tmp_chrom[0]
            ), f"Chromosomes in {pfile_path}.pvar are not the same as in {pfile_list[0]}"

    ##################################
    # run admix-simu
    ##################################
    phgeno_list = [
        f"-POP{i + 1} {tmp_dir}/{os.path.basename(pfile)}.phgeno"
        for i, pfile in enumerate(pfile_list)
    ]

    admix_dat = [
        "\t".join(
            [str(n_indiv * 2), "ADMIX", *[f"POP{i}" for i in np.arange(1, n_pop + 1)]]
        ),
        "\t".join([str(n_gen), "0", *[str(prop) for prop in admix_prop]]),
    ]

    dat_file = os.path.join(tmp_dir, "admix.dat")
    with open(dat_file, "w") as f:
        f.writelines("\n".join(admix_dat))

    # cat dat_file
    print("\n".join(admix_dat))
    df_snp_info.insert(1, "chrom", chrom)
    df_snp_info.insert(
        2,
        "M",
        interpolate_genetic_position(
            chrom=chrom, pos=df_snp_info["position"], build=build
        ),
    )
    print(df_snp_info.head())
    snp_file = os.path.join(tmp_dir, "snp_info.txt")
    df_snp_info.to_csv(snp_file, sep="\t", index=False, header=False)

    cmd = " ".join(
        [
            os.path.join(admix_simu_dir, "simu-mix.pl"),
            dat_file,
            snp_file,
            os.path.join(tmp_dir, "admix"),
            *phgeno_list,
        ]
    )
    admix.logger.info(cmd)
    subprocess.check_output(cmd, shell=True)

    cmd = " ".join(
        [
            os.path.join(admix_simu_dir, "bp2anc.pl"),
            os.path.join(tmp_dir, "admix.bp"),
            ">",
            os.path.join(tmp_dir, "admix.hanc"),
        ]
    )
    admix.logger.info(cmd)
    subprocess.check_output(cmd, shell=True)

    ##################################
    # post-processing
    ##################################

    shutil.move(os.path.join(tmp_dir, "admix.hanc"), f"{out_prefix}.hanc")
    shutil.move(os.path.join(tmp_dir, "admix.phgeno"), f"{out_prefix}.phgeno")
    tmp.cleanup()