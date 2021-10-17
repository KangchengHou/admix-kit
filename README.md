# admix-tools
![python package](https://github.com/KangchengHou/admix-tools/actions/workflows/workflow.yml/badge.svg)
![document](https://github.com/KangchengHou/admix-tools/actions/workflows/sphinx.yml/badge.svg)

See [![](https://img.shields.io/badge/docs-latest-blue.svg)](https://kangchenghou.github.io/admix-tools) 
for documentation.

## Install
```
git clone https://github.com/KangchengHou/admix-tools
cd admix-tools
pip install -r requirements.txt; pip install -e .
```

## File formats
- We use PLINK2 .pgen format to store the (potentially phased) genotype
- [TODO] we use customized .lanc format to store the local ancestry file.
- These two file formats allow efficient storage of genotype file (by the design of PLINK2), while allowing for fast random access of the genotype.
- Also, all functions of PLINK2 can be used for additional analyses.
- [TODO] zarr format for the auxilary information? such minor allele frequencies and LD scores

## .lanc format
The first line contains meta-information <n_indiv> <n_snp> <n_anc> then <n_indiv> lines to follow:
For each line, we record the information of the break points: 
<anc_0><anc_1>:<pos> <anc_0>
The <anc_0><anc_1> are the ordered (according the phase) local ancestry information

An example of the file looks like
```
3 300 2
01:100 00:300
10:120 01:300
00:300
```
Note that these ranges are [start, stop) and index starts from 300. Also, the last position 
of each line should ends with <n_snp>. We provide helper function to convert between this 
sparse file format and dense matrix format. See XX and XX functions.

## Data structures
- With admix-kit, we use a admix.Dataset to support various convenient operations for manipulating data sets.
- admix.Dataset stores (n_snp, n_indiv, n_ploidy) genotype matrix and local ancestry matrix.
- dset = admix.Dataset 
- dset.snp stores SNP-level covariates
- dset.indiv stores indiv-level covariates
- dset.snpm stores SNP-level matrix, such as LD matrix.
- dset.indivm stores individual-level matrix, such as GRM matrix.
- dset.geno, dset.lanc are on-disk dask arrays.
- dset.loc[snp_subset, indiv_subset] select subset of SNPs and/or individuals

## Functions
We have a set of functions 

## TODO
- Complete GWAS demonstration and pipelines.
- Add helper function to take vcf file and RFmix local ancestry file and spit out zarr.
- First release: complete pipeline for simulating and estimating genetic correlation.