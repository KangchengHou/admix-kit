# Admixed genotype simulation
We describe the pipeline to simulate genotypes of admixed individuals using reference ancestral populations (such as those in 1,000 Genomes project).

## Overview
In the following, we go through each part of the pipeline using an example of simulating individuals with African-European genetic ancestries using CEU, YRI as reference populations. In details, we will
1. download 1,000 Genomes project reference.
2. decide choices of SNP set because you may want to save time/memory to simulate only HapMap3 SNPs. Currently our pipeline only supports simulating one chromosome at a time. To extend simulations to the whole genome, users can repeat runs for 22 chromosomes (X chromosome is not supported yet).
3. determine the proportion of ancestral populations, number of generations, and number of admixed individuals to simulate.

## Optional: download software and reference data
We use many 3rd party software in this pipeline. You may want to download them in advance in case you don't have internet connection in running the code.
```bash
for name in plink2 hapmap3_snps; do
    admix download-dependency --name ${name}
done

admix download-dependency --name genetic_map --build hg38
admix download-dependency --name genetic_map --build hg19
```
OK to skip this step as `admix-kit` will automatically download the software and reference data if you have the internet connection.

## Simulation
```bash
BUILD=hg38 # genome build to use
CHROM=22 # chromosome to simulate
OUT_DIR="data/example_data" # output directory

```

```bash
# download 1,000 Genomes reference panel from plink2 website 
# (https://www.cog-genomics.org/plink/2.0/resources)
# you can use your own reference data (but you need to format them into plink2 pgen format)
# see https://www.cog-genomics.org/plink/2.0/input

admix get-1kg-ref --dir data/1kg-ref-${BUILD} --build ${BUILD}

mkdir -p ${OUT_DIR} # create a directory to store simulated data

# subset hapmap3 SNPs in chromosome 22 to save time/memory
admix subset-hapmap3 \
    --pfile data/1kg-ref-${BUILD}/pgen/all_chr \
    --build ${BUILD} \
    --chrom ${CHROM} \
    --out ${OUT_DIR}/hm3_chrom${CHROM}.snp

plink2 \
    --pfile data/1kg-ref-${BUILD}/pgen/all_chr \
    --extract ${OUT_DIR}/hm3_chrom${CHROM}.snp \
    --make-pgen \
    --out ${OUT_DIR}/1kg-ref

# Simulate 3-way admixture
admix haptools-simu-admix \
    --pfile ${OUT_DIR}/1kg-ref \
    --admix-prop '{"CEU": 0.4, "YRI": 0.1, "PEL": 0.5}' \
    --pop-col Population \
    --mapdir data/1kg-ref-${BUILD}/metadata/genetic_map/ \
    --n-gen 15 \
    --n-indiv 1000 \
    --out ${OUT_DIR}/CEU-YRI-PEL

# Simulate 2-way admixture
admix haptools-simu-admix \
    --pfile ${OUT_DIR}/1kg-ref \
    --admix-prop '{"CEU": 0.2, "YRI": 0.8}' \
    --pop-col Population \
    --mapdir data/1kg-ref-${BUILD}/metadata/genetic_map/ \
    --n-gen 10 \
    --n-indiv 10000 \
    --out ${OUT_DIR}/CEU-YRI

# you will obtain 
# (1) plink2 phased genotype: data/simulated-{CEU-YRI-PEL|CEU-YRI}.{pgen,pvar,psam}
# (2) local ancestry: data/simulated-{CEU-YRI-PEL|CEU-YRI}.lanc
```

These simulated datasets can be downloaded with `admix.dataset.download_simulated_example_data()`.

We perform several analyses using these example datasets in the following notebooks:

- [Basic statistics and visualization](notebooks/analyze-admix-simu-data.ipynb).
- [Association testing (GWAS)](notebooks/assoc.ipynb).
- [Genetic correlation across local ancestry segments](notebooks/genet-cor.ipynb).


<!-- ## Simulate with HAPGEN2
```
admix subset-pop-indiv \
    --pfile data/example_data/1kg-ref \
    --pop ASW \
    --out data/example_data/1kg-ref-ASW.indiv

plink2 --pfile data/example_data/1kg-ref \
    --keep data/example_data/1kg-ref-ASW.indiv \
    --make-pgen \
    --out data/example_data/1kg-ref-ASW

admix hapgen2 \
    --pfile data/example_data/1kg-ref-ASW \
    --n-indiv 1000 \
    --chrom 22 \
    --build hg38 \
    --out data/example_data/1kg-ref-ASW-hapgen2
``` -->