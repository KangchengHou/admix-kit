"""
Standalone functions corresponding to the external tools
"""

import admix
import tempfile
import os
import subprocess
import pandas as pd
import shutil
from . import get_dependency, get_cache_data


def hapgen2(
    pfile: str,
    chrom: int,
    n_indiv: int,
    out_prefix: str,
    genetic_map: str = "hg38",
    plink_kwargs: dict = None,
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

    # *
    # convert PLINK2 to legend and hap
    # *

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

    # *
    # simulate genotype with HAPGEN2
    # *

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
        " -no_gen_output",
    ]
    subprocess.check_call(" ".join(cmd_hapgen2), shell=True)

    # *
    # convert HAPGEN2 to PLINK2
    # *

    cmd_convert_pgen = [
        f"--haps {tmp_data_prefix}.hapgen2.controls.haps.gz",
        f"--legend {tmp_data_prefix}.hapgen2.legend {chrom}",
        "--make-pgen",
        f"--out {out_prefix}",
    ]

    admix.tools.plink2.run(" ".join(cmd_convert_pgen))

    shutil.move(f"{tmp_data_prefix}.hapgen2.gz.summary", f"{out_prefix}.hapgen2.log")

    # *
    # clean up
    # *
    tmp.cleanup()