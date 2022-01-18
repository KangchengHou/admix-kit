"""
admix.data is for all sorts of data manipulation
including genotype matrix, local ancestry matrix

These functions should not depend on admix.Dataset, but rather can be used
on their own alone.
"""

from ._geno import allele_per_anc, af_per_anc, geno_mult_mat
from ._geno import calc_snp_prior_var, impute_with_mean
from ._stats import quantile_normalize, pval2chisq, lambda_gc, zsc2pval
from ._lanc import Lanc, concat_lancs, lanc_impute_single_chrom, haplo2diplo
from ._utils import index_over_chunks, make_dataset, match_prs_weights
from ._misc import convert_dummy