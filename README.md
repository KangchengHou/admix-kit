# admix-kit
![python package](https://github.com/KangchengHou/admix-tools/actions/workflows/workflow.yml/badge.svg)
[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://kangchenghou.github.io/admix-kit)

`admix-kit` is a Python library to faciliate analyses and methods development of genetics data from admixed populations. Jump to [Quick start (CLI)](#quick-start-command-line-interface) or [Quick start (Python)](#quick-start-python-api).

> `admix-kit` is still in beta version, we welcome any [feedbacks](https://github.com/KangchengHou/admix-kit/pulls) and [bug reports](https://github.com/KangchengHou/admix-kit/issues).   

## Install
```bash
# Install admix-kit with Python 3.7, 3.8, 3.9
git clone https://github.com/KangchengHou/admix-kit
cd admix-kit
pip install -r requirements.txt; pip install -e .
```

> Installation requires cmake version > 3.12. Use `cmake --version` to check your cmake version. Use `pip install cmake` to install the latest version.

## File formats
- `.pgen|.psam|.pvar`: PLINK2 format phased genotype to easily manipulate data (using PLINK2) and fast random access within python.
- `.lanc`: customized local ancestry matrix format  (see [.lanc format](#lanc-format) below)
- `.snp_info`: SNP information file, such as allele frequency.
- `.indiv_info`: individual information file, such as top PCs.

### .lanc format
`.lanc` is a text file containing a matrix of local ancestry of shape `<n_snp> x <n_indiv> x <2 ploidy>`. 

The first line contains two numbers: `<n_snp>` for number of SNPs and `<n_indiv>` for number of indivduals. Then `<n_indiv>` lines follow with each line corresponds to one individual:
For each line, the local ancestry change points are recorded as
`<pos>:<anc1><anc2>` which records the position of the change point and the *ordered* ancestries (according to the phase) local ancestry information.

An example of `.lanc` file will make the format clear:
```
300 3
100:01 300:00
120:10 300:01
300:00
```
This corresponds to a 300 SNPs x 3 individuals x 2 ploidy matrix. The corresponding dense matrix for the first individual can be reconstructed using the following code:
```python
# example for the first individual in the above example file
break_list = [100, 300]
anc0_list = [0, 0]
anc1_list = [1, 0]    
start = 0
lanc = np.zeros(300, 2, dtype=np.int8)
for stop, anc0, anc1 in zip(break_list, anc0_list, anc1_list):
    lanc[start : stop, 0] = anc0
    lanc[start : stop, 1] = anc1
    start = stop
```

Note these ranges are right-open intervals `[start, stop)` and the last position of each line always ends with `<n_snp>`. We provide helper function to convert between sparse `.lanc` format and dense matrix format.


## Quick start (command line interface)
We can perform local ancestry inference, simple phenotype simulation and association testing using the `admix`
command line interface.
```bash
# install the admix-kit package, make a new directory and cd into it
# try running the following code

# copy test data
# the test data is built from ASW, CEU and YRI individuals 1,000 Genome projects 
# see tests/test-data/make-toy.sh for scripts to build the toy data
test_data_dir=$(python -c "import admix; print(admix.dataset.get_test_data_dir())")
cp ${test_data_dir}/toy-* ./

# rename the provided .lanc as we are to compute this now
mv toy-admix.lanc toy-admix.old.lanc

# local ancestry inference for ASW individuals with CEU and YRI individuals
admix lanc \
    --pfile toy-admix \
    --ref-pfile toy-all \
    --ref-pop-col "Population" \
    --ref-pops "CEU,YRI" \
    --out toy-admix.lanc

# quantitative phenotype simulation 
# toy-admix.pheno (simulated phenotype) and toy-admix.beta (simulated effects) 
# will be generated
admix simulate-quant-pheno \
    --pfile toy-admix \
    --hsq 0.5 \
    --n-causal 2 \
    --n-sim 2 \
    --seed 1234 \
    --out-prefix toy-admix

# extract PC1, PC2 of the toy-admix.indiv_info as toy-admix.covar
awk '{print $1, $3, $4}' toy-admix.indiv_info > toy-admix.covar

# association testing of individual SNPs of simulated phenotype
# several NA values are expected due to some SNP has zero frequency in this toy dataset
admix assoc \
    --pfile toy-admix \
    --pheno toy-admix.pheno \
    --pheno-col SIM0 \
    --covar toy-admix.covar \
    --method ATT,TRACTOR \
    --out toy-admix.assoc
```

## Quick start (Python API)
**Note that `admix-kit` is in development and python API is subject to change. You may prefer command line interface which is currently more stable. Meanwhile, any suggestion / bug report and pull request is welcome.**

Central in python API is the `admix.Dataset` class, which support various convenient operations for manipulating large on-disk data sets.
Have a try on the following example in Jupyter notebook.
```python
import admix

# load toy data
dset = admix.dataset.load_toy_admix()

# overview of data set
dset

# SNP attributes, CHROM, POS, REF, ALT, etc.
dset.snp

# individual attributes
dset.indiv

# phased genotype (n_snp, n_indiv, 2)
dset.geno

# local ancestry (n_snp, n_indiv, 2)
dset.lanc

# subset the first 50 SNPs
dset[0:50, :]

# subset the first 50 individuals
dset[:, 0:50]

# subset the first 50 SNPs and first 50 individuals
dset[0:50:, 0:50]

# calculate allele per ancestry backgrounds
dset.allele_per_anc()

# calculate allele frequencies per ancestry backgrounds
dset.af_per_anc()
```

## admix.Dataset
admix.Dataset have dset.snp attributes, and is assumed to contain 
- index `snp` (`ID` in the file)
- chromosomes `CHROM` (without `chr` prefix)
- positions `POS`
- reference allele `REF`
- alternative allele `ALT` (we assume bi-allelic variants, use plink2 --biallelic to filter the data beforehand.)

PLINK2 format [.pvar](https://www.cog-genomics.org/plink/2.0/formats#pvar) will have these attributes.
