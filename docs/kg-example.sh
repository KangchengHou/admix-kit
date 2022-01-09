# We use the example of studying admixed individuals from ASW group.
# Local ancestry inference is performed with

# Setup
OLD_DIR=$PWD
DATA_DIR=kg-example

mkdir -p ${DATA_DIR}
cd ${DATA_DIR} || exit

# Download 1kg plink2 files from https://www.cog-genomics.org/plink/2.0/resources#1kg_phase3
wget https://www.dropbox.com/s/asn2ehkkwh4vjhr/all_phase3_ns.pgen.zst?dl=1 -O raw.pgen.zst
wget https://www.dropbox.com/s/ud2emtq98z257wn/all_phase3_ns_noannot.pvar.zst?dl=1 -O raw.pvar.zst
wget https://www.dropbox.com/s/yozrzsdrwqej63q/phase3_corrected.psam?dl=1 -O raw.psam

plink2 --zst-decompress raw.pgen.zst >raw.pgen
plink2 --zst-decompress raw.pvar.zst >raw.pvar

# use awk to filter IDs based on column 6 (either CEU, YRI or ASW)
awk '{if ($6=="CEU" || $6=="YRI" || $6=="ASW") print $1}' raw.psam >tmp_indiv.txt

# ALL includes CEU, YRI and ASW
plink2 --pfile raw \
    --rm-dup exclude-all \
    --max-alleles 2 \
    --maf 0.01 \
    --exclude-snp . \
    --keep tmp_indiv.txt \
    --snps-only \
    --chr 1-22 \
    --make-pgen --out ALL

# clean up
rm raw.*
rm tmp_indiv.txt

# PCA and local ancestry inference are all based on pruned set of SNPs
mkdir -p pruned
# prune SNPs in LD
admix prune --pfile ALL --out pruned/ALL

# PCA
admix pca --pfile pruned/ALL --out pruned/ALL.pca

# Visualize the data in PC space
admix plot-pca \
    --pfile pruned/ALL \
    --label-col Population \
    --pca pruned/ALL.pca.eigenvec \
    --out pruned/ALL.pca.png

# Extract the admixed individuals (ASW group)
awk '{if ($6=="ASW") print $1}' ALL.psam >ADMIX.indiv
plink2 --pfile pruned/ALL \
    --keep ADMIX.indiv \
    --make-pgen --out pruned/ADMIX

# For simplicity, now we use a subset of the data from chromosome 21, 22
# when adapting this script to your own data, you can use the whole data
# by changing the following line to CHROM_LIST=$(seq 1 22)

# Perform local ancestry inference
for chrom in $(seq 21 22); do
    # subset chromosome `chrom`
    plink2 --pfile pruned/ADMIX \
        --chr "${chrom}" \
        --make-pgen --out pruned/ADMIX."${chrom}"

    plink2 --pfile pruned/ALL \
        --chr "${chrom}" \
        --make-pgen --out pruned/ALL."${chrom}"

    admix lanc \
        --pfile pruned/ADMIX."${chrom}" \
        --ref-pfile pruned/ALL."${chrom}" \
        --ref-pop-col "Population" \
        --ref-pops "CEU,YRI" \
        --out pruned/ADMIX."${chrom}".lanc
done

# Now we have obtained the local ancestry of ASW individuals
# We can use the following command to compile the data set for later analysis.

mkdir -p compiled
for chrom in $(seq 21 22); do
    # subset pgen for chromosome `chrom`
    plink2 --pfile ALL \
        --chr "${chrom}" \
        --keep ADMIX.indiv \
        --make-pgen --out compiled/ADMIX."${chrom}"

    # impute local ancestry for the full data set
    admix lanc-impute \
        --pfile compiled/ADMIX."${chrom}" \
        --ref-pfile pruned/ADMIX."${chrom}" \
        --out compiled/ADMIX."${chrom}".lanc
done

# Now we have compiled the data set for later analysis.
# Let's start with several analyses.

# Simulating phenotypes

admix simulate-pheno \
    --pfile "compiled/ADMIX.21" \
    --hsq 0.01 \
    --n-causal 2 \
    --seed 1234 \
    --out-prefix compiled/ADMIX \
    --family binary

# Association testing using ATT or Tractor
admix assoc \
    --pfile "compiled/ADMIX.21" \
    --pheno "compiled/ADMIX.pheno" \
    --pheno-col "SIM0" \
    --out compiled/SIM_0.assoc \
    --family binary \
    --method TRACTOR,ATT

cd "${OLD_DIR}" || exit
