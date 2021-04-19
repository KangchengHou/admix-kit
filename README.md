# admix-tools

## Install
```
git clone https://github.com/KangchengHou/admix-tools
cd admix-tools
pip install -r requirements.txt; pip install -e .
```

## Data format
- `.lanc`: local ancestry file (n_indiv * 2, n_snp)
- `.hap`: haplotype file (n_indiv * 2, n_snp)
- `.geno`: genotype file (n_indiv, n_snp)
- `.legend`: SNP information, deliminated by space.
- `.sample`: Sample information
- `.pheno`: phenotype file
TODO: We provide utility function to transform common file type into these format.


## Development guide
- All source code is in `admix` folder. 
- We have several modules (1) ancestry (2) data (3) finemap (4) plot (5) simulate (6) utils. By default, each module should be .py file. However, when one module is becoming too complex, we may make it a folder with a `__init__.py` exporting all the functions.