# Simulate admixed genotypes
We describe the pipeline to simulate genotypes of admixed individuals using reference ancestral populations (such as those in 1,000 Genomes project).
```{note}
`admix-kit` is required to run the pipeline. Refer to [this page](install.md) to install `admix-kit`.
```

## Overview
In the following, we go through each part of the pipeline using an example of simulating individuals with African-European genetic ancestries using CEU, YRI as reference populations. In details, we will
1. download 1,000 Genomes project reference.
2. decide choices of SNP set because you may want to save time/memory to simulate only HapMap3 SNPs. Currently our pipeline only supports simulating one chromosome at a time. To extend simulations to the whole genome, users can repeat runs for 22 chromosomes (X chromosome is not supported yet).
3. determine the proportion of ancestral populations, number of generations, and number of admixed individuals to simulate. (If you want to simulate N admixed individuals, you need to simulate N individuals using HAPGEN2 for each ancestral populations.)

```{note}
Run in Linux environment (because HAPGEN2 can only be run in Linux). 
```
## Step 1: prepare ancestral reference population data
We go through the following steps:
1. download 1,000 Genomes reference data 
2. subset HapMap3 SNPs from chromosome 22 
3. subset CEU, YRI as ancestral populations in order to simulate admixed individuals with 20% European and 80% African ancestries (similar to African African individuals).

```bash
# genome build to use
BUILD=hg38
# number of admixed individuals to simulate
N_INDIV=1000
# chromosome 
CHROM=22
# number of generations
N_GEN=8
```

```bash
# download 1,000 Genomes reference panel from plink2 website 
# (https://www.cog-genomics.org/plink/2.0/resources)
# you can use your own reference data (but you need to format them into plink2 pgen format)
# see https://www.cog-genomics.org/plink/2.0/input

admix get-1kg-ref --dir data/1kg-ref-${BUILD} --build ${BUILD}

mkdir -p data/ancestry

# subset hapmap3 SNPs
admix subset-hapmap3 \
    --pfile data/1kg-ref-${BUILD}/pgen/all_chr \
    --build ${BUILD} \
    --chrom ${CHROM} \
    --out data/ancestry/hm3_chrom${CHROM}.snp

# subset individuals
for pop in CEU YRI PEL; do
    admix subset-pop-indiv \
        --pfile data/1kg-ref-${BUILD}/pgen/all_chr \
        --pop ${pop} \
        --out data/ancestry/${pop}.indiv
done

# subset plink2
for pop in CEU YRI PEL; do
    plink2 --pfile data/1kg-ref-${BUILD}/pgen/all_chr \
        --keep data/ancestry/${pop}.indiv \
        --extract data/ancestry/hm3_chrom${CHROM}.snp \
        --make-pgen \
        --out data/ancestry/${pop}
done
```

## Step 2: Extend ancestral populations using HAPGEN2
Next, we use HAPGEN2 to extend the ancestral populations. As aim to simulate 1000 admixed individuals, we simulate 1,000 individuals in each ancestral population.

```bash
for pop in CEU YRI PEL; do
    admix hapgen2 \
        --pfile data/ancestry/${pop} \
        --chrom ${CHROM} \
        --n-indiv ${N_INDIV} \
        --out data/ancestry/${pop}.hapgen2 \
        --build ${BUILD}
done
```

## Step 3: Simulate admixture process using `admix-simu`
We use [admix-simu](https://github.com/williamslab/admix-simu) to simulate the admixture process of ancestral populations.
We simulate 8 generations, with admixture proportion of 20% / 80%.

### Simulation
```bash
# admix-simu should be automatically installed using this command
admix admix-simu \
    --pfile-list "['data/ancestry/CEU.hapgen2', 'data/ancestry/YRI.hapgen2', 'data/ancestry/PEL.hapgen2']" \
    --admix-prop "[0.4,0.1,0.5]" \
    --n-indiv ${N_INDIV} \
    --n-gen ${N_GEN} \
    --build ${BUILD} \
    --out data/admix

# you will obtain 
# (1) plink2 phased genotype: data/admix.{pgen,pvar,psam}
# (2) local ancestry: data/admix.lanc
```

<!-- TODO: To perform basic check on the simulated data. We performed some basic analyses below. -->