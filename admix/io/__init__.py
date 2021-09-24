from ._read import read_gcta_grm
from ._read import read_digit_mat, write_digit_mat, read_vcf

from ._write import write_gcta_grm

__all__ = [
    "read_gcta_grm",
    "write_gcta_grm",
    "read_vcf",
]
