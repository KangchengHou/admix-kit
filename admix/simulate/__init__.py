from ._pheno import quant_pheno, quant_pheno_1pop, quant_pheno_grm
from ._pheno import binary_pheno, sample_case_control
from ._lanc import hap_lanc, calculate_mosaic_size
from ._geno import admix_geno, admix_geno_simple

# from ._genotype import simulate_hap

__all__ = [
    "quant_pheno",
    "binary_pheno",
    "sample_case_control",
    "quant_pheno_1pop",
    "quant_pheno_grm",
    "admix_geno",
]
