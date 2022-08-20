# Guideline to use RFmix for local ancestry inference

```{warning}
This guideline assume you have **phased** genotype in PLINK2 pgen format of your own dataset.
PLINK2 can convert many other formats to pgen format [(see here)](https://www.cog-genomics.org/plink/2.0/input). 
If you need help, [raise an issue](https://github.com/KangchengHou/admix-kit/issues) so we can help 
update this guideline to adapt to your data.
```

## Step 1: download and process reference data

```bash
admix get-1kg-ref --dir=/path/to/1kg
```

```{note}
This script will (1) download plink2 pgen files for 1,000 genomes project (2) perform
basic QCs (3) download metadata of sample map and genetic maps. This will take a while to process (~2hrs).
You can also [read more details](download-1kg-detail.md) behind the scene.
```

As RFmix takes reference dataset in vcf format, we convert pgen format to vcf format

```bash
# convert plink2 to vcf
root_dir=/path/to/1kg/
mkdir -p ${root_dir}/1kg/
for chrom in {1..22}; do
    plink2 \
    --pfile ${root_dir}/pgen/all_chr \
    --export vcf bgz \
    --chr ${chrom} \
    --out ${root_dir}/vcf/chr${chrom} && tabix -p vcf ${root_dir}/vcf/chr${chrom}.vcf.gz
done
```

## Step 1.5 (optional) subset admixed individuals using joint PCA with 1KG reference

We first merge the 1kg data and sample data into a single file and perform a joint PCA.
```bash
ref_pfile=/path/to/1kg/all_chr    # path to 1kg pgen file (all chromosomes)
sample_pfile=/path/to/sample/all_chr # path to sample pgen file (all chromosomes)
out_dir=/path/to/joint-pca    # path to output directory

# merge 1kg dataset and sample dataset
admix pfile-merge-indiv \
    --pfile1 ${ref_pfile} \
    --pfile2 ${sample_pfile} \
    --out ${out_dir}/merged

# perform PCA
plink2 --bfile ${out_dir}/merged \
    --pca approx \
    --out ${out_dir}/merged_pca
```

Then we plot the joint PCA results.
```bash
admix plot-joint-pca \
    --ref-pfile ${ref_pfile} \
    --pca-prefix ${out_dir}/merged_pca \
    --out ${out_dir}/merged_pca
```

And we select admixed individuals based on PC.
```bash
admix select-admix-indiv \
    --ref-pfile ${ref_pfile} \
    --pca-prefix ${out_dir}/merged_pca \
    --superpop1 EUR --superpop2 AFR \
    --exclude-pop2 ASW,ACB \
    --out ${out_dir}/selected_admix
```

And create new plink2 files
```bash
for chrom in $(seq 1 22); do
    plink2 \
        --pfile /path/to/sample/chr${chrom} \
        --keep ${out_dir}/selected_admix.indiv \
        --make-pgen --out /path/to/sample/admix/chr${chrom} \
        --maf 0.005 \
        --memory 24000
done
```

## Step 2: run RFmix

### Step 2.1: download and compile RFmix

Follow `https://github.com/slowkoni/rfmix#building-rfmix` to build the rfmix executables.

### Step 2.2: run RFmix
Now we will run RFmix to infer local ancestry for your genotype file (see how to prepare the genotype file in [Prepare dataset section](prepare-dataset.md)).
```bash
## Specify constants
chrom=XX # this script should be ran 1 chromosome at a time
build=hg38 # replace hg38 with hg19 as needed
REF_DIR=/path/to/1kg/ # see step 1, set it to the root directory containing metadata/ pgen/ vcf/
RFMIX=/path/to/rfmix_exe # see above to download and compile rfmix executables

pfile=/path/to/your-plink2-file # see "Prepare dataset" section
out_prefix=/path/to/output/chr${chrom} # output prefix

## convert pfile to vcf because RFmix takes vcf as input
plink2 \
    --pfile ${pfile} \
    --chr ${chrom} \
    --output-chr 26 \
    --export vcf bgz \
    --out ${out_prefix}.tmp

tabix -p vcf ${out_prefix}.tmp.vcf.gz

## Filter reference samples
# extract 1st column (sample name) and 2nd column (populations)
# Here, we use CEU, YRI, PEL for reference populations for Latino populations
# NOTE: it is up to your choice what reference populations to use
# e.g., For local ancestry inference of European-African admixed individuals, we 
# recommend using CEU, YRI only.
awk '$2=="CEU" || $2=="YRI" || $2=="PEL" {print $1 "\t" $2}' \
    ${REF_DIR}/metadata/${build}.unrelated_sample.tsv >${out_prefix}.tmp.sample_map.tsv

## run RFmix
${RFMIX} \
    -f ${out_prefix}.tmp.vcf.gz \
    -r ${REF_DIR}/vcf/${build}.chr${chrom}.vcf.gz \
    -m ${out_prefix}.tmp.sample_map.tsv \
    -g ${REF_DIR}/metadata/genetic_map/${build}.chr${chrom}.tsv \
    --chromosome=${chrom} \
    -o ${out_prefix}

## Remove temporary files
# .sis file corresponds to XX
# .fb file corresponds to XX
rm ${out_prefix}*tmp*
rm ${out_prefix}.sis.tsv
rm ${out_prefix}.fb.tsv
```

After you finish this step, you can proceed to the rest of [preparing the dataset](prepare-dataset.md).