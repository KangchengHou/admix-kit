# admix-tools

## Install
```
git clone https://github.com/KangchengHou/admix-tools
cd admix-tools
pip install -r requirements.txt; pip install -e .
```

## Data format
We recommend using Zarr to manage the dataset. Each Zarr file will have the following components.
- `lanc`: local ancestry file (n_indiv, n_snp, 2) 
- `hap`: haplotype file (n_indiv, n_snp, 2)
- `allele_per_anc`: allele counts per ancestry (n_indiv, n_snp, n_anc)

TODO: We provide utility function to transform common file type into these format.

We use text files to store other files.
This tool is processing the matrix. For convenience, we just use the following internally. We use zarr to store the results.
- lanc: (n_indiv, n_snp, 2)
- hap: (n_indiv, n_snp, 2)

## Development guide
- All source code is in `admix` folder. 
- We have several modules (1) ancestry (2) data (3) finemap (4) plot (5) simulate (6) utils. 
  By default, each module should be .py file. However, when one module is becoming too complex, 
  we may make it a folder with a `__init__.py` exporting all the functions.
  
## TODO