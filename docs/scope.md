# SCOPE analysis

## Merge SCOPE with 1,000 Genomes project ancestral populations
```{note}
Before this merging, you need to prepare the 1,000 genomes project in PLINK2 format.
See [RFmix section](rfmix.md) for more details to download 1,000 genomes project data.
```

```bash
# [optional] filter SNPs
# MAF Filter
# plink2 --pfile ../format-data/out/hm3/merged --maf 0.01 --make-pgen --out scope/maf_01
# admix prune --pfile scope/maf_01 --out scope/maf_01_pruned

OUT_DIR=scope/
mkdir -p $OUT_DIR
# 1. align the SNPs in 1kg and sample pfiles
admix pfile-align-snp \
    --pfile1 /u/project/pasaniuc/kangchen/DATA/new-plink2-1kg/out/pgen/hg38 \
    --pfile2 ../format-data/out/hm3/merged \
    --out ${OUT_DIR}/merged

# rename files
for suffix in pgen pvar psam log; do
    mv ${OUT_DIR}/merged.1.$suffix ${OUT_DIR}/1kg.$suffix
done

for suffix in pgen pvar psam log; do
    mv ${OUT_DIR}/merged.2.$suffix ${OUT_DIR}/sample.$suffix
done

# 2. calculate population-specific frequencies
admix pfile-freq-within \
    --pfile ${OUT_DIR}/1kg \
    --group-col SuperPop \
    --groups AFR,EAS,EUR,SAS \
    --out ${OUT_DIR}/1kg_freq 
```

Follow [SCOPE github](https://github.com/sriramlab/SCOPE) to compile the software.

```bash
# 3. run SCOPE
SCOPE_BIN=/u/project/pasaniuc/kangchen/software/SCOPE/build/scope

# SCOPE requires plink1 format
plink2 --pfile scope/sample \
    --make-bed \
    --out scope/sample

# k must match number of ancestries in the frequency file
${SCOPE_BIN} \
    -g scope/sample \
    -k 4 \
    -freq scope/1kg_freq.frq.strat \
    -o supervised_ 
```

