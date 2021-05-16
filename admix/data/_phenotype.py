from pysnptools.snpreader import Pheno
import numpy as np
import pandas as pd

class Phenotype(object):
    """
    representes a phenotype,
    usually associated with a PLINK file
    """
    
    def __init__(self, path, indiv_subset=None):
        self.path = path
        pheno = Pheno(path)
        self.data = pd.concat(
            [pd.DataFrame(pheno.iid.astype(str), columns=['FID', 'IID']),
            pd.DataFrame(pheno.read().val, columns=pheno.col.astype(str))], axis=1)
        self.data = self.data.replace(-9, np.nan)
        if indiv_subset is not None:
            # because we have FID + IID
            subset_indiv = pd.DataFrame(indiv_subset, columns=['FID', 'IID'])
            self.data = pd.merge(self.data, subset_indiv, on=['FID', 'IID'])
    
    def to_file(self, path):
        self.data.to_csv(path, sep='\t', float_format='%.6f', index=False, na_rep='-9')