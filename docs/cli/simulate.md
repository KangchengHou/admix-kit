# Simulation
In this section, we demonstrate a complete simulation pipeline. Starting from reference populations in 1,000 Genomes, we simulate genotypes, phenotypes for ancestral and admixed populations.

## Prepare source data
```bash
# genome build to use
BUILD=hg19
# number of admixed individuals to simulate
N_INDIV=100000
# chromosome 
CHROM=22
# number of generations
N_GEN=8

# setup
REF_DATA_DIR=data/1kg-ref-${BUILD}
RAW_DATA_DIR=data/raw
GENO_DATA_DIR=data/geno
mkdir -p ${RAW_DATA_DIR}
mkdir -p ${GENO_DATA_DIR}
```

```bash
admix get-1kg-ref --dir ${REF_DATA_DIR} --build ${BUILD}

# subset hapmap3 SNPs
admix subset-hapmap3 \
    --pfile ${REF_DATA_DIR}/pgen/all_chr \
    --build ${BUILD} \
    --chrom ${CHROM} \
    --out ${RAW_DATA_DIR}/hm3_chrom${CHROM}.snp

# subset individuals
for pop in CEU YRI; do
    admix subset-pop-indiv \
        --pfile ${REF_DATA_DIR}/pgen/all_chr \
        --pop ${pop} \
        --out ${RAW_DATA_DIR}/${pop}.indiv
done

# subset plink2
for pop in CEU YRI; do
    plink2 --pfile ${REF_DATA_DIR}/pgen/all_chr \
        --keep ${RAW_DATA_DIR}/${pop}.indiv \
        --extract ${RAW_DATA_DIR}/hm3_chrom${CHROM}.snp \
        --make-pgen \
        --out ${RAW_DATA_DIR}/${pop}
done
```

## Extend ancestral populations using HAPGEN2

```bash
for pop in CEU YRI; do
    admix hapgen2 \
        --pfile ${RAW_DATA_DIR}/${pop} \
        --chrom ${CHROM} \
        --n-indiv ${N_INDIV} \
        --out ${GENO_DATA_DIR}/${pop} \
        --build ${BUILD}
done
```

## Simulate admixed individuals
```bash
# admix-simu should be automatically installed using this command
admix admix-simu \
    --pfile-list "['../data/geno/CEU', '../data/geno/YRI']" \
    --admix-prop "[0.2,0.8]" \
    --n-indiv ${N_INDIV} \
    --n-gen ${N_GEN} \
    --build ${BUILD} \
    --out ../data/geno/admix
```

## Rename individuals
```python
import dapgen
import shutil
for pop in ["admix", "CEU", "YRI"]:
    df_psam = dapgen.read_psam(f"{pop}.psam").reset_index()
    # rename old files to backup
    shutil.move(f"{pop}.psam", f"{pop}.psam.bak")
    df_psam["#IID"] = [f"{pop}{i}" for i in range(len(df_psam))]
    df_psam[["#IID"]].to_csv(f"{pop}.psam", index=False, header=True)
```


## Simulate phenotypes
Simulate beta first
Then simulate environmental noise and phenotype.

```{eval-rst}
.. autofunction:: admix.cli.simulate_pheno
```
