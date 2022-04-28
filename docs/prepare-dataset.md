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

## Step 1: format genotype

### Step 1.1 (optional): select well-imputed SNPs
Often we start with the imputed genotype from imputation server. We can filter by MAF > 0.005 (5th column) and R2 > 0.8 (7th column) to select the SNPs with high quality.
```bash
IN_DIR=/path/to/vcf
OUT_DIR=/path/to/imputed

# filter well-imputed SNPs
zcat ${IN_DIR}/chr${chrom}.info.gz | awk 'NR>1 {if($5>0.005 && $7>0.8) print $1}' > \
    ${OUT_DIR}/chr${chrom}.snplist

# convert to PLINK2 format
plink2 --vcf ${IN_DIR}/chr${chrom}.dose.vcf.gz \
    --extract ${OUT_DIR}/chr${chrom}.snplist \
    --rm-dup exclude-all \
    --snps-only \
    --maf 0.005 \
    --max-alleles 2 \
    --make-pgen \
    --memory 16000 \
    --out ${OUT_DIR}/chr${chrom}

# (alternative) if your vcf file is already processed, use the following
plink2 --vcf ${vcf} --make-pgen --out ${out_plink}
```

PLINK2 is also versatile for converting other formats into .pgen format. See more at [https://www.cog-genomics.org/plink/2.0/input#pgen](https://www.cog-genomics.org/plink/2.0/input#pgen).

### Step 1.2 (optional): select HM3 SNPs
Most genetic analysis (e.g., local ancestry inference) can be made more efficient by subsetting the data to HapMap3 SNPs.
```bash
admix subset-hapmap3 --pfile ${imputed_pfile} --out ${hm3_pfile} --build hg38
```

```{note}
Make sure your source data is phased because it is essential for many analyses with admix-kit. Use `plink2 --pfile <pfile> --pgen-info` for basic check. If there is a line "Explicitly phased hardcalls present", that means phasing data is present.
```

## Step 2: Local ancestry inference
There are many choices for local ancestry inference. We assume that you have performed the local ancestry. We provide helper function to convert the local ancestry results into .lanc format ([see more details below](#lanc)) which is a compact format for storing local ancestry.

To convert the RFmix local ancestry into .lanc format, use the following command.
This command can be applied to both imputed and hm3 data.
```bash
admix lanc-convert \
    --pfile <pgen_prefix> \      # e.g., dset.chr1
    --rfmix <rfmix_msp_path> \  # e.g., rfmix/dset.chr1.msp.tsv
    --out <lanc_path>           # e.g., dset.chr1.lanc
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