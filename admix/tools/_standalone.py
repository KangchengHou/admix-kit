"""
Standalone functions corresponding to the external tools
"""

import admix
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
    n_indiv: int,
    out_prefix: str,
    genetic_map: str,
    chrom: int = None,
    plink_kwargs: dict = dict(),
):
    """simulate genotype with HAPGEN2

    Parameters
    ----------
    pfile : str
        path to PLINK2 pfile
    genetic_map : str
        genetic map, downloaded from
        https://alkesgroup.broadinstitute.org/Eagle/downloads/tables/
    chrom : int
        chromosome number for the plink2 file. The file can only contain data from one chromosome,
        if not specified, the function will try to guess the chromosome number from the plink2 file
    n_indiv : int
        Number of individuals to simulate
    plink_kwargs : dict
        kwargs to pass to plink2, e.g. --from-bp XX --to-bp YY
    out_prefix : str
        output prefix
    """
    tmp_dir = out_prefix + ".hapgen2tmpdata"
    assert not os.path.exists(
        tmp_dir
    ), f"{tmp_dir} should not exist, please remove it before running this function"
    os.makedirs(tmp_dir, exist_ok=False)
    tmp_data_prefix = os.path.join(tmp_dir, "hapgen2_data")

    # fetch chromosome
    if chrom is None:
        chrom = dapgen.read_pvar(pfile + ".pvar")["CHROM"].values
        assert (
            len(np.unique(chrom)) == 1
        ), "only one chromosome is allowed in the plink2 file"
        chrom = chrom[0]

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

    assert genetic_map in [
        "hg19",
        "hg38",
    ], f"genetic map {genetic_map} not supported, only hg19 and hg38 are supported"

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
    shutil.rmtree(tmp_dir)


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

    Notes
    -----
    When a given position is out of the range of the genetic map, the genetic position
    will be extrapolated linearly.
    """
    assert build in ["hg19", "hg38"]
    from scipy.interpolate import interp1d

    df_map = pd.read_csv(
        admix.tools.get_cache_data("genetic_map", build=build), delim_whitespace=True
    )
    df_map = df_map[df_map["chr"] == chrom]
    assert np.all(np.sort(df_map["position"]) == df_map["position"])
    interp = interp1d(
        x=df_map["position"],
        y=df_map["Genetic_Map(cM)"],
        assume_sorted=True,
        fill_value="extrapolate",
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
    out_prefix: str
        output prefix
    admix_simu_dir: str
        Directory to the admix_simu software

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
    assert build in ["hg19", "hg38"], "build should be hg19 or hg38"
    assert (
        n_gen > 1
    ), "n_gen should be greater than 1, otherwise admix-simu will run with errors"
    assert len(pfile_list) == len(
        admix_prop
    ), "`pfile_list`, `admix_prop` should have the same length"
    n_pop = len(pfile_list)

    for i, pfile in enumerate(pfile_list):
        admix.logger.info(f"POP{i}={pfile}")

    admix_simu_dir = admix.tools.get_dependency("admix-simu")
    ##################################
    # format input
    ##################################
    tmp_dir = out_prefix + ".admixsimutmpdata"
    assert not os.path.exists(
        tmp_dir
    ), f"{tmp_dir} should not exist, please remove it before running this function"
    os.makedirs(tmp_dir, exist_ok=False)
    df_snp_info = None
    chrom = None
    for pfile_path in pfile_list:
        pfile = os.path.basename(pfile_path)
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
    # insert Morgan genetic position (cM / 100)
    df_snp_info.insert(
        2,
        "M",
        interpolate_genetic_position(
            chrom=chrom, pos=df_snp_info["position"], build=build
        )
        / 100,
    )
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

    ## convert phgeno to pgen
    df_snp_info[["id", "position", "a0", "a1"]].to_csv(
        os.path.join(tmp_dir, "admix.legend"), sep=" ", index=False
    )
    subprocess.check_output(
        f"cat {tmp_dir}/admix.phgeno | sed 's/./& /g' > {tmp_dir}/admix.haps",
        shell=True,
    )
    admix.tools.plink2.run(
        f"--haps {tmp_dir}/admix.haps --legend {tmp_dir}/admix.legend {chrom} "
        + f"--make-pgen --out {tmp_dir}/admix",
    )
    for suffix in ["pgen", "pvar", "psam", "log"]:
        shutil.move(os.path.join(tmp_dir, f"admix.{suffix}"), f"{out_prefix}.{suffix}")

    ## convert local ancestry
    lanc = admix.data.read_bp_lanc(os.path.join(tmp_dir, "admix.bp"))
    lanc.write(f"{out_prefix}.lanc")
    shutil.move(os.path.join(tmp_dir, "admix.hanc"), f"{out_prefix}.hanc")
    shutil.move(os.path.join(tmp_dir, "admix.bp"), f"{out_prefix}.bp")

    # clean up
    shutil.rmtree(tmp_dir)

    # hint the file path
    admix.logger.info(f"Output files are saved to {out_prefix}.*")


def haptools_simu_admix(
    pfile: str,
    admix_prop: List[float],
    pop_col: str,
    mapdir: str,
    n_gen: int,
    n_indiv: int,
    out_prefix: str,
    seed=1234,
):
    """Wrapper for haptools simgenotype

    Parameters
    ----------
    pfile : str
        list of input prefixes
    admix_prop :
        proportion of admixed individuals relative proportion of the admixture,
        will be rescaled to 1.
    n_gen : int
        number of generations to simulate
    n_indiv : int
        number of individuals to simulate
    out_prefix: str
        output prefix
    """
    from bisect import bisect_left

    ##################################
    # check input
    ##################################
    assert (
        n_gen > 1
    ), "n_gen should be greater than 1, otherwise admix-simu will run with errors"

    ##################################
    # format input
    ##################################
    tmp_dir = out_prefix + ".tmpdata"
    assert not os.path.exists(
        tmp_dir
    ), f"{tmp_dir} should not exist, please remove it before running this function"
    os.makedirs(tmp_dir, exist_ok=False)

    ##################################
    # run haptools
    ##################################

    admix_dat = [
        "\t".join([str(n_indiv), "ADMIX", *[pop for pop in admix_prop]]),
        "\t".join([str(n_gen), "0", *[str(admix_prop[pop]) for pop in admix_prop]]),
    ]

    dat_file = os.path.join(tmp_dir, "admix.dat")
    with open(dat_file, "w") as f:
        f.writelines("\n".join(admix_dat))

    # cat dat_file
    print("\n".join(admix_dat))

    sample_df = dapgen.read_psam(pfile + ".psam")
    sample_info_file = os.path.join(tmp_dir, "sample_info.txt")
    sample_df[[pop_col]].to_csv(sample_info_file, sep="\t", index=True, header=True)

    snp_df = dapgen.read_pvar(pfile + ".pvar")
    chrom = np.unique(snp_df["CHROM"])
    assert len(chrom) == 1, "Only one chromosome is allowed in the plink2 file"
    start, stop = np.min(snp_df["POS"]), np.max(snp_df["POS"])

    cmds = [
        "haptools simgenotype",
        f"--model {dat_file}",
        f"--mapdir {mapdir}",
        f"--ref_vcf {pfile}.pgen",
        f"--sample_info {sample_info_file}",
        f"--out {tmp_dir}/admix.pgen",
        f"--region {chrom[0]}:{start}-{stop}",
        f"--seed {seed}",
    ]
    cmd = " ".join(cmds)
    admix.logger.info(cmd)
    subprocess.check_output(cmd, shell=True)

    ##################################
    # post-processing
    ##################################

    anc_map = {pop: order for order, pop in enumerate(admix_prop)}

    with open(os.path.join(tmp_dir, "admix.bp")) as f:
        lines = f.readlines()

    seps = list(np.where([line.startswith("Sample_") for line in lines])[0]) + [
        len(lines)
    ]

    breaks = []
    values = []
    for i in range(len(seps) - 1):
        haplo = lines[seps[i] + 1 : seps[i + 1]]

        hap_pos = []
        hap_values = []
        for line in haplo:
            anc, chrom, pos, cm = line.strip().split()
            anc_order = anc_map[anc]
            hap_pos.append(int(pos))
            hap_values.append(anc_order)
        hap_breaks = [bisect_left(snp_df.POS.values, p) for p in hap_pos]
        breaks.append(hap_breaks)
        values.append(hap_values)

    dip_breaks, dip_values = admix.data.clean_lanc(
        *admix.data.haplo2diplo(breaks=breaks, values=values),
        remove_repeated_val=True,
    )
    lanc = admix.data.Lanc(breaks=dip_breaks, values=dip_values)
    lanc.write(f"{out_prefix}.lanc")

    for suffix in ["pgen", "pvar", "psam", "bp"]:
        shutil.move(os.path.join(tmp_dir, f"admix.{suffix}"), f"{out_prefix}.{suffix}")

    # clean up
    shutil.rmtree(tmp_dir)

    # hint the file path
    admix.logger.info(f"Output files are saved to {out_prefix}.*")
