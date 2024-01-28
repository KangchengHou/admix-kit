from typing import List
from ._dataset import Dataset, is_aligned
from ._load import (
    load_toy,
    load_toy_admix,
    load_ukb_eur_afr_hm3,
    get_test_data_dir,
    download_simulated_example_data,
)

# required columns in dset.snp
REQUIRED_SNP_COLUMNS = ["CHROM", "POS", "REF", "ALT"]
