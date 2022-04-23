import admix
import subprocess
import dapgen
import os
import glob
import numpy as np
import pandas as pd
from ._utils import log_params


def lanc(
    pfile: str,
    ref_pfile: str,
    ref_pop_col: str,
    ref_pops: str,
    out: str,
):
    log_params("lanc", locals())

    sample_dset = admix.io.read_dataset(pfile=pfile)
    ref_dset = admix.io.read_dataset(pfile=ref_pfile)

    assert set(sample_dset.snp.index) == set(ref_dset.snp.index), (
        "`pfile` and `ref_pfile` must have the same snp index"
        "(snp match feature coming soon)."
    )

    ref_dsets = [
        ref_dset[:, (ref_dset.indiv[ref_pop_col] == pop).values] for pop in ref_pops
    ]
    est = admix.ancestry.lanc(sample_dset=sample_dset, ref_dsets=ref_dsets)
    admix.data.Lanc(array=est).write(out)


def lanc_count(lanc: str, out: str, n_anc: int = None):
    """Count the number / proportion of local ancestries for each individual

    Parameters
    ----------
    lanc : str
        path to the lanc file, this can be a .lanc file, a wildcard of .lanc files,
        or a directory containing .lanc files. If the corresponding .psam file is
        present, the .psam file will be used as the individual list.
    out : str
        path to the output file
    n_anc : int
        number of ancestral populations in the data
    """
    log_params("lanc-count", locals())
    if lanc.endswith(".lanc"):
        lanc_path = [lanc]
    elif "*" in lanc:
        lanc_path = glob.glob(lanc)
    elif os.path.isdir(lanc):
        lanc_path = [p for p in glob.glob(lanc + "/*.lanc")]
    else:
        raise ValueError("Unable to parse lanc pathname")

    admix.logger.info(f"Found {len(lanc_path)} lanc files: {','.join(lanc_path)}")
    # read psam if available
    psam_path = [p.replace(".lanc", ".psam") for p in lanc_path]

    if all(os.path.exists(p) for p in psam_path):
        # check all psam files have the same individual ID
        psam_indiv = [dapgen.read_psam(p).index for p in psam_path]
        assert all(
            psam_indiv[0].equals(i) for i in psam_indiv[1:]
        ), "Individuals in psam files do not match"
        indiv_list = psam_indiv[0].values
    elif not any(os.path.exists(p) for p in psam_path):
        indiv_list = None
    else:
        raise ValueError("either .psam all exists or none exists")

    lanc_mat = admix.data.Lanc(lanc_path[0])
    n_indiv = lanc_mat.n_indiv
    if indiv_list is not None:
        assert n_indiv == len(
            indiv_list
        ), "Number of individuals in lanc and psam files do not match"
    else:
        indiv_list = np.arange(n_indiv).astype(str)

    lanc_count = lanc_mat.lanc_count()
    if n_anc is not None:
        assert (
            lanc_count.shape[1] == n_anc
        ), "Number of ancestral populations do not match"
    else:
        n_anc = lanc_count.shape[1]
        admix.logger.info(f"Inferred number of ancestral populations: {n_anc}")

    for p in lanc_path[1:]:
        lanc_count += admix.data.Lanc(p).lanc_count(n_anc=n_anc)

    lanc_prop = lanc_count / lanc_count.sum(axis=1, keepdims=True)
    admix.logger.info(f"Writing lanc count file: {out}")

    count_cols = [f"COUNT{i + 1}" for i in range(n_anc)]
    prop_cols = [f"PROP{i+1}" for i in range(n_anc)]
    df_res = pd.DataFrame(
        data=np.concatenate([lanc_count, lanc_prop], axis=1),
        index=indiv_list,
        columns=count_cols + prop_cols,
    )

    df_res[count_cols] = df_res[count_cols].astype(int)
    df_res.index.name = "indiv"
    df_res.to_csv(out, sep="\t", float_format="%.4g")


def lanc_convert(pfile: str, out: str, rfmix: str = None, raw: str = None):
    """Convert local ancestry inference results (e.g. RFmix .msp.tsv) to a .lanc file

    Parameters
    ----------
    pfile : str
        Path to the pfile. The path is without the .pgen suffix
    out : str
        Path to the output file
    rfmix : str
        Path to the rfmix .msp.tsv file,
    raw : str
        Path to the raw file
    """
    log_params("lanc-convert", locals())

    # only one of rfmix and raw should be specified
    assert (rfmix is None) + (
        raw is None
    ) == 1, "Only one of rfmix and raw should be specified"
    if rfmix is not None:
        geno, df_snp, df_indiv = dapgen.read_pfile(pfile, phase=True)
        admix.logger.info(f"Reading rfmix file: {rfmix}")
        lanc = admix.io.read_rfmix(
            path=rfmix,
            df_snp=df_snp,
            df_indiv=df_indiv,
        )
        admix.logger.info(f"Obtaining local ancestry {lanc}")
        admix.logger.info(f"Writing lanc file: {out}")
        lanc.write(out)

    if raw is not None:
        assert False, "raw not implemented yet"


def lanc_impute(pfile: str, ref_pfile: str, out: str = None):
    """Impute the local ancestry for `pfile` using `ref_pfile`

    Parameters
    ----------
    pfile : str
        Path to the pfile
    ref_pfile : str
        Path to the reference pfile
    out : str
        Path to the output pfile (default to pfile + ".lanc")
    """
    log_params("lanc-impute", locals())

    # check <pfile>.lanc does not exist
    assert not os.path.exists(pfile + ".lanc"), "`pfile` already has a .lanc file"

    sample_dset = admix.io.read_dataset(pfile=pfile)
    ref_dset = admix.io.read_dataset(pfile=ref_pfile)
    ref_lanc = admix.data.Lanc(ref_pfile + ".lanc")

    sample_lanc = ref_lanc.impute(
        ref_dset.snp[["CHROM", "POS"]].values, sample_dset.snp[["CHROM", "POS"]].values
    )
    if out is None:
        out = pfile + ".lanc"
    assert not os.path.exists(out), f"out={out} already exists"
    sample_lanc.write(out)


def lanc_rfmix(
    sample_vcf: str,
    ref_vcf: str,
    sample_map: str,
    genetic_map: str,
    out_prefix: str,
    chrom: int = None,
    rfmix_path: str = "rfmix",
):
    """Estimate local ancestry from a sample vcf and reference vcf

    TODO: Contents in https://kangchenghou.github.io/admix-tools/en/main/prepare_data.html
        should be subsumed into this function
    Parameters
    ----------
    pfile : str
        PLINK2 pfile for admixed individuals
    ref_pfile : str
        PLINK2 pfile for reference individuals
    sample_map : str
        Text file with two column containing the population of individuals in ref_vcf
        the unique population will be used as reference ancestral population in
        estimation.
    genetic_map: str
        Text file with two column containing the genetic distance between two
    out_prefix: str
        Prefix for the output files.
    method : str, optional
        method for estimating local ancestry, by default "rfmix"

    """
    log_params("lanc-rfmix", locals())

    # Step 1: use bcftools to align the sample and reference vcf
    align_ref_code = (
        f"""
        sample_vcf={sample_vcf}
        ref_vcf={ref_vcf}
        out_prefix={out_prefix}
    """
        + """

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
    """
    )

    print(align_ref_code)
    subprocess.check_call(align_ref_code, shell=True)

    # Step 2: use rfmix to estimate local ancestry
    rfmix_code = (
        f"""
        sample_vcf={out_prefix}.sample.vcf.gz
        ref_vcf={out_prefix}.ref.vcf.gz
        sample_map={sample_map}
        genetic_map={genetic_map}
        chrom={chrom}
        out_prefix={out_prefix}
        rfmix={rfmix_path}
        """
        + """
        ${rfmix} \
            -f ${sample_vcf} \
            -r ${ref_vcf} \
            -m ${sample_map} \
            -g ${genetic_map} \
            --chromosome=${chrom} \
            -o ${out_prefix}
        """
    )

    print(rfmix_code)
    subprocess.check_call(rfmix_code, shell=True)
