# admix-kit
![python package](https://github.com/KangchengHou/admix-tools/actions/workflows/workflow.yml/badge.svg)
[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://kangchenghou.github.io/admix-kit)

`admix-kit` is a Python library to faciliate analyses and methods development of genetics data from admixed population. Jump to [Quick start (CLI)](#quick-start-command-line-interface) or [Quick start (Python)](#quick-start-python-api).

## Install
```
git clone https://github.com/KangchengHou/admix-kit
cd admix-kit
pip install -r requirements.txt; pip install -e .
```

## File formats
- `.pgen|.psam|.pvar`: PLINK2 format phased genotype to easily manipulate data (using PLINK2) and fast random access within python.
- `.lanc`: customized local ancestry matrix format  (see below, TODO: add link)
- `.snp_info`: SNP information file, such as allele frequency.
- `.indiv_info`: individual information file, such as top PCs.

### .lanc format
`.lanc` is a text file containing a matrix of local ancestry of shape `<n_snp> x <n_indiv> x <2 ploidy>`. 

The first line contains two numbers: `<n_snp>` for number of SNPs and `<n_indiv>` for number of indivduals. Then `<n_indiv>` lines follow with each line corresponds to one individual:
For each line, the local ancestry change points are recorded as
`<pos>:<anc1><anc2>` which records the position of the change point and the *ordered* ancestries (according to the phase) local ancestry information.

Here is an example of `.lanc` file
```
300 3
100:01 300:00
120:10 300:01
300:00
```
The local ancestry dense matrix can be reconstructed using the following procedure:
```python
# example for the first individual in the above example file
break_list = [100, 300]
anc0_list = [0, 0]
anc1_list = [1, 0]    
start = 0
for stop, anc0, anc1 in zip(break_list, anc0_list, anc1_list):
    lanc[start : stop, 0] = anc0
    lanc[start : stop, 1] = anc1
    start = stop
```

Note these ranges are right-open intervals `[start, stop)` and the last position of each line always ends with `<n_snp>`. We provide helper function to convert between this sparse file format and dense matrix format.


## Quick start (command line interface)
We perform local ancestry inference, 
```bash
# copy test data
test_data_dir=$(python -c "import admix; print(admix.dataset.get_test_data_dir())")
cp ${test_data_dir}/toy-* ./

# rename the provided .lanc as we are to compute this now
mv toy-admix.lanc toy-admix.old.lanc

# local ancestry inference
admix lanc \
    --pfile toy-admix \
    --ref-pfile toy-all \
    --ref-pop-col "Population" \
    --ref-pops "CEU,YRI" \
    --out toy-admix.lanc

# phenotype simulation, 
# toy-admix.pheno (simulated phenotype) and toy-admix.beta (simulated effects) 
# will be generated
admix simulate-quant-pheno \
    --pfile toy-admix \
    --hsq 0.05 \
    --n-causal 2 \
    --n-sim 2 \
    --seed 1234 \
    --out-prefix toy-admix

# association testing for the simulated phenotype
admix assoc-quant \
    --pfile toy-admix \
    --pheno toy-admix.pheno \
    --pheno-col SIM0 \
    --method ATT,TRACTOR \
    --out toy-admix.assoc
```



## Quick start (Python API)
**Note that `admix-kit` is in development and python API is subject to change. If this is a concern, please only use command line interface (which is more stable[TODO: add link to full documentation] for now.**

**At the same time, any suggestion / bug report and pull requests are welcome.**

```python
import admix

# load genetic data and local ancestry seperately
dset = admix.dataset.load_
dset.snp

dset.indiv

dset.geno

dset.lanc

dset subset

dset.allele_per_anc()

dset.

```

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

## Acknowledgement
TODO: