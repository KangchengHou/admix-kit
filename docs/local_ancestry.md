# Local ancestry inference
rfmix -f cohort.vcf -r AFR_EUR.phase3_shapeit2_mvncall_integrated_v5a.20130502.chr$i.vcf.gz --chromosome=${i} -m AFR_EUR.1kg_order.indivs.pops.txt -g HapMapcomb_genmap_chr${i}.txt -e 1 -n 5 -o cohort.rfmix.chr$i

```bash
for i in {1..22}; do \
python RunRFMix.py \
-e 2 \
-w 0.2 \
--num-threads 4 \
--use-reference-panels-in-EM \
--forward-backward \
PopPhased \
CEU_YRI_ACB_chr${i}.alleles \
CEU_YRI_ACB.classes \
CEU_YRI_ACB_chr${i}.snp_locations \
-o CEU_YRI_ACB_chr${i}.rfmix; done
```