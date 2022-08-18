# Simulate individual-level genotypes for admixed populations
This pipeline is designed to simulate genotypes of admixed individuals using reference ancestral populations (such as those in 1,000 Genomes project).

## Overview
1. Download 1,000 Genomes project reference.
2. Decide choices of SNP set because you may want to save some time to simulate only HapMap3 SNPs. Currently our pipeline only supports simulation one chromosome at a time. To extend simulations to the whole genome, users can repeat the following run repeatedly for 22 chromosomes (X chromosome is not supported yet).
3. Determine the proportion of ancestral populations, number of generations, and number of admixed individuals to simulate. If you want to simulate N admixed individuals, you need to simulate N individuals using HAPGEN2 for each ancestral populations.

## Step 1: prepare ancestral reference population data
In step 1, we (a) download 1,000 Genomes reference data (b) subset HapMap3 SNPs from chromosome 22 (c) subset CEU, YRI as ancestral populations in order to simulate admixed individuals with 20% European and 80% African ancestries (similar to African African individuals).

```bash
# genome build to use
BUILD=hg19
# number of admixed individuals to simulate
N_INDIV=1000
# chromosome 
CHROM=22
# number of generations
N_GEN=8
```

```bash
# download 1,000 Genomes reference panel (this step will take 2-3 hours)
admix get-1kg-ref --dir data/1kg-ref --build ${BUILD}

# subset hapmap3 SNPs
admix subset-hapmap3 \
    --pfile data/1kg-ref/pgen/all_chr \
    --build ${BUILD} \
    --chrom ${CHROM} \
    --out data/ancestry/hm3_chrom${CHROM}.snp

# subset individuals
for pop in CEU YRI; do
    admix subset-pop-indiv \
        --pfile data/1kg-ref/pgen/all_chr \
        --pop ${pop} \
        --out data/ancestry/${pop}.indiv
done

# subset plink2
for pop in CEU YRI; do
    plink2 --pfile data/1kg-ref/pgen/all_chr \
        --keep data/ancestry/${pop}.indiv \
        --extract data/ancestry/hm3_chrom${CHROM}.snp \
        --make-pgen \
        --out data/ancestry/${pop}
done
```

## Step 2: Extend ancestral populations using HAPGEN2
In step 2, we use HAPGEN2 to extend the ancestral populations. We aim to simulate 1000 admixed individuals, therefore we simulate 1,000 individuals for each ancestral population.

```bash
for pop in CEU YRI; do
    admix hapgen2 \
        --pfile data/ancestry/${pop} \
        --chrom ${CHROM} \
        --n-indiv ${N_INDIV} \
        --out data/ancestry/${pop}.hapgen2 \
        --build ${BUILD}
done
```

## Step 3: Simulate admixture process using admix-simu
`admix-simu` can be downloaded from https://github.com/williamslab/admix-simu

```bash
# download and compile admix-simu
git clone https://github.com/williamslab/admix-simu.git
cd admix-simu && make && cd ..

```

```bash
# ADMIX_SIMU_DIR is the path to the admix-simu directory (that is git cloned above)
ADMIX_SIMU_DIR=./admix-simu
admix admix-simu \
    --pfile-list "['data/ancestry/CEU.hapgen2','data/ancestry/YRI.hapgen2']" \
    --admix-prop "[0.2,0.8]" \
    --n-indiv ${N_INDIV} \
    --n-gen ${N_GEN} \
    --build ${BUILD} \
    --out data/simu/admix \
    --admix-simu-dir ${ADMIX_SIMU_DIR}
```