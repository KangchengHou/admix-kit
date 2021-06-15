# Extract the typed SNPs from a VCF file and align with reference panel. Output the
# aligned sample vcf and reference vcf files.
#
# `sample_vcf` is assumed to be imputed and phased. the code is tailored for VCF files
#   obtained from TOPMed imputation server but should be straightforward to adapt to
#   data obtained with other procedures.
# Both `sample_vcf` and `ref_vcf` should be indexed with tabix -p vcf ${vcf}
#
# params:
# sample_vcf: vcf file containing genotypes for samples
# ref_vcf: vcf file containing genotypes for references, e.g., 1000 Genomes
# out_prefix: prefix to the output, ${out_prefix}.sample.vcf.gz and
#   ${out_prefix}.ref.vcf.gz will be generated

sample_vcf=$1
ref_vcf=$2
out_prefix=$3

tmp_dir=${out_prefix}.tmp/

mkdir ${tmp_dir}

if [[ ! -f ${sample_vcf}.tbi ]]; then
    echo "${sample_vcf} is not indexed. Please index it with tabix. Exiting..."
    exit
fi

# match reference panel
bcftools isec -n =2 ${sample_vcf} ${ref_vcf} -p ${tmp_dir} -c none

cat ${tmp_dir}/0000.vcf | bgzip -c >${tmp_dir}/sample.vcf.gz
cat ${tmp_dir}/0001.vcf | bgzip -c >${tmp_dir}/ref.vcf.gz

# remove chr
for i in {1..22}; do
    echo "chr$i $i" >>${tmp_dir}/chr_name.txt
done

bcftools annotate --rename-chrs ${tmp_dir}/chr_name.txt ${tmp_dir}/sample.vcf.gz |
    bgzip >${out_prefix}.sample.vcf.gz
bcftools annotate --rename-chrs ${tmp_dir}/chr_name.txt ${tmp_dir}/ref.vcf.gz |
    bgzip >${out_prefix}.ref.vcf.gz

# clean up
rm -rf ${tmp_dir}
