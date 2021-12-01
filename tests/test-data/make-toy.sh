# Generate toy data
# This data is generated using admix-kit version
# https://github.com/KangchengHou/admix-kit/tree/573cdaef5d481ba7995ea40a66f783c419f176d1
# 1. Make PLINK2 dataset.
# 2. Infer local ancestry with LAMPLD.
# 3. Simulate phenotypes.
# 4. Perform association testing.

# 1. Make PLINK2 dataset.
wget https://www.dropbox.com/s/ozraccaavbtdkzm/chr22_phase3.pgen.zst?dl=1 -O raw.pgen.zst
wget https://www.dropbox.com/s/g5sucurqv46y6q9/chr22_phase3_noannot.pvar.zst?dl=1 -O raw.pvar.zst
wget https://www.dropbox.com/s/yozrzsdrwqej63q/phase3_corrected.psam?dl=1 -O raw.psam

plink2 --zst-decompress raw.pgen.zst >raw.pgen
plink2 --zst-decompress raw.pvar.zst >raw.pvar

rm *.zst

python make-toy.py subset-indiv

plink2 --pfile raw \
    --rm-dup exclude-all \
    --max-alleles 2 \
    --thin-count 10000 \
    --maf 0.01 \
    --snps-only \
    --keep indiv.txt \
    --seed 0 \
    --make-pgen --out toy-all

rm raw.p*
rm indiv.txt
rm toy-all.log

python make-toy.py process

# clean up
rm toy-admix.log
