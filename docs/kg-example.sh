# We use the example of studying admixed individuals from ASW group.
# Local ancestry inference is performed with

# Setup
OLD_DIR=$PWD
DATA_DIR=kg-example

mkdir -p ${DATA_DIR}
cd ${DATA_DIR}

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
    --keep indiv.txt \
    --snps-only \
    --chr 1-22 \
    --make-pgen --out ALL

# clean up
rm raw.*
rm tmp_indiv.txt

# Visualize the data in PC space
admix pca --pfile ALL.pruned --out ALL.pca

admix plot-pca \
    --pfile ALL \
    --label-col Population \
    --pca ALL.pca.eigenvec \
    --out ALL.pca.png

# Perform local ancestry inference

cd ${OLD_DIR}
