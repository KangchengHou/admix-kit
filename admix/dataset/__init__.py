"""
Sept. 24, 2021: this file is not in any use. After other modules are mature enough,
    we can devote effort to design a Dataset class.
"""

from typing import List
from ._dataset import Dataset, is_aligned
from ._load import load_toy, load_toy_admix, load_ukb_eur_afr_hm3, get_test_data_dir

# required columns in dset.snp
REQUIRED_SNP_COLUMNS = ["CHROM", "POS", "REF", "ALT"]
