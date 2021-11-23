"""
Generating toy data

1. Make PLINK2 dataset.
2. Infer local ancestry with LAMPLD.
3. Simulate phenotypes.
4. Perform association testing.
"""
import numpy as np
import pandas as pd
import dapgen
import fire
import admix


def subset_indiv():
    df_sample = dapgen.read_psam("raw.psam")
    df_sample = df_sample[df_sample.Population.isin(["CEU", "YRI", "ASW"])]

    with open("indiv.txt", "w") as f:
        f.writelines("\n".join(df_sample.index))


def infer_lanc():
    """
    Infer local ancestry
    """