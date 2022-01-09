# Prepare dataset

## Overview
To start genetic analysis for admixed populations with admix-kit, it is required to compile the data set into the following files.

- Phased genotype in PLINK2 format.
- Local ancestry inference results .lanc.
- Additional individuals' information .indiv_info
- Additional SNPs' information .snp_info

For genome-wide analysis, it is recommended to divide the data set by chromosomes. For example, the file structure will look like
```
dset.chr1.pgen dset.chr1.psam dset.chr1.pvar dset.chr1.lanc dset.chr1.snp_info dset.chr1.indiv_info
dset.chr2.pgen dset.chr2.psam ....
```

We details the steps to prepare data set as follows:

## PLINK2 format genotype

To convert phased vcf file into PLINK2 format, use the following command:
```bash
plink2 --vcf <vcf_path> --make-pgen --out <plink2_path>
```
PLINK2 is also versatile for converting other formats into .pgen format. See more at [https://www.cog-genomics.org/plink/2.0/input#pgen](https://www.cog-genomics.org/plink/2.0/input#pgen).

```{note}
Make sure your source data is phased because it is essential for many analyses with admix-kit.
```

## Local ancestry inference
There are many choices for local ancestry inference. We assume that you have performed the local ancestry. We provide helper function to convert the local ancestry results into .lanc format ([see more details below](#lanc)) which is a compact format for storing local ancestry.

### RFmix
```bash
admix convert-lanc \
    --plink <plink_path> \      # e.g., dset.chr1
    --rfmix <rfmix_msp_path> \  # e.g., rfmix/dset.chr1.msp.tsv
    --out <lanc_path>           # e.g., dset.chr1.lanc
```

### Raw matrix format
```bash
admix convert-lanc \
    --plink <plink_path> \
    --raw <rfmix_msp_path> \ # TODO: implement the raw function
    --out <lanc_path>
```


## Other files
PLINK2 genotype file and `.lanc` file are almost you need to start the analysis. The 
other files might be useful for better structuring your analysis. `.snp_info` contains 
SNP information file, such as allele frequency, and `.indiv_info` contains individual 
information file, such as top PCs.

## Processing and analyzing an example data set
See [kg-example.sh](https://github.com/KangchengHou/admix-kit/blob/main/docs/kg-example.sh) 
for an example of preparing data set as well as a simple analysis: first simulate a 
phenotype and then perform GWAS.

## Details of file formats
### .pgen|.psam|.pvar
These files 
### .lanc
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