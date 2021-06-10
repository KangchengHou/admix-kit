"""
Data structures
===========================
We show the central data stuctures used in this package.

In the analysis of individuals from admixed population, we store (1) genotypes, (2) local ancestries
(3) information about SNPs (4) information about each individuals.

These data will be of shape

#. genotype (n_snp, n_indiv, n_haploid)
#. local ancestry (n_snp, n_indiv, n_haploid)
#. information about SNPs (n_snp, [])
#. information about individuals (n_indiv, [])

We make use of `xarray` to deal with all the data in an unified way, specifically, we will have an xarray.Dataset to
store these. Below is a typical structure.
"""

import xarray as xr
from admix.simulate import simulate_hap, simulate_lanc, simulate_continuous_phenotype
from admix.plot import plot_local_anc
import matplotlib.pyplot as plt

n_indiv = 100
n_snp = 1000

hap = simulate_hap(n_indiv, n_snp)
lanc = simulate_lanc(n_indiv, n_snp, 100)

print(hap.shape)
print(lanc.shape)
