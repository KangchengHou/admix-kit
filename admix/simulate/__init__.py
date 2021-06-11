from ._phenotype import (
    sample_case_control,
    # simulate_phenotype_case_control_1snp,
    continuous_phenotype,
    continuous_phenotype_grm,
)
from ._lanc import simulate_lanc
from ._genotype import simulate_hap

__all__ = [
    "sample_case_control",
    # "simulate_phenotype_case_control_1snp",
    "continuous_phenotype",
    "continuous_phenotype_grm",
    "simulate_lanc",
    "simulate_hap",
]
