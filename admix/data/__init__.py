"""
all about admix.Dataset
"""

from ._utils import make_dataset, load_toy, load_lab_dataset

from ._read import read_digit_mat, write_digit_mat, read_vcf

__all__ = [
    "read_digit_mat",
    "write_digit_mat",
    "compute_allele_per_anc",
    "load_toy",
    "load_lab_dataset",
    "make_dataset",
    "Dataset",
]
