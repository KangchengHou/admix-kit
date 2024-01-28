"""
Load existing data sets
"""
from typing import List
from os.path import dirname, join
import numpy as np
import os
import xarray as xr
from tqdm import tqdm
import pandas as pd
import warnings
from ._dataset import Dataset
import admix


def get_test_data_dir() -> str:
    """
    Get toy dataset directory

    Returns
    -------
    str
        Toy dataset directory
    """
    test_data_path = join(dirname(__file__), "../../tests/test-data")
    return test_data_path


def download_simulated_example_data(dir=None) -> Dataset:
    """
    Load example data set

    Returns
    -------
    Dataset
    """
    import urllib
    import zipfile

    if dir is None:
        dir = "./"

    # confirm dir/example_data does not exist
    if os.path.exists(os.path.join(dir, "example_data")):
        admix.logger.info(
            f"Example data set already exists at {dir}/example_data, skip downloading"
        )
    else:
        urllib.request.urlretrieve(
            "https://github.com/KangchengHou/admix-kit/releases/download/v0.1.2/example_data.zip",
            os.path.join(dir, "example_data.zip"),
        )
        with zipfile.ZipFile(os.path.join(dir, "example_data.zip"), "r") as f:
            f.extractall(dir)

        admix.logger.info(f"Example data set downloaded to {dir}/example_data")

        # remove the zip file
        os.remove(os.path.join(dir, "example_data.zip"))


def load_toy_admix() -> Dataset:
    """
    Load toy admixed data set with African and European ancestries

    Returns
    -------
    Dataset
    """
    dset = admix.io.read_dataset(join(get_test_data_dir(), "toy-admix"), n_anc=2)
    return dset


def load_toy() -> List[Dataset]:
    """Load toy dataset

    Load simulated
    (1) 50 admixed individuals
    (2) 50 EUR individuals
    (3) 50 AFR individuals

    5000 SNPs

    Returns
    -------
    List[admix.Dataset]
        [dset_admix, dset_eur, dset_afr]
    """

    # TODO: change the data format, use .pgen and .lanc
    import xarray as xr

    module_path = dirname(__file__)
    test_data_path = join(module_path, "../../tests/test-data")
    dset_eur = xr.open_zarr(join(test_data_path, "eur.zip"))
    dset_afr = xr.open_zarr(join(test_data_path, "afr.zip"))
    dset_admix = xr.open_zarr(join(test_data_path, "admix.zip"))

    dset_list = [
        Dataset(
            geno=np.swapaxes(dset_admix.geno.data, 0, 1),
            lanc=np.swapaxes(dset_admix.lanc.data, 0, 1),
            n_anc=2,
            indiv=dset_admix.indiv.to_dataframe().drop(columns=["indiv"]),
            snp=dset_admix.snp.to_dataframe().drop(columns=["snp"]),
        )
    ]
    for dset in [dset_eur, dset_afr]:
        dset_list.append(
            Dataset(
                geno=np.swapaxes(dset.geno.data, 0, 1),
                n_anc=1,
                indiv=dset.indiv.to_dataframe().drop(columns=["indiv"]),
                snp=dset.snp.to_dataframe().drop(columns=["snp"]),
            ),
        )

    return dset_list


"""
load compiled data sets
"""


# PAGE admixed individuals with EUR - AFR ancestries in hapmap3 density
# def load_page_eur_afr_hm3(
#     chrom=None,
#     GENO_DIR="/u/project/pasaniuc/pasaniucdata/admixture/projects/PAGE-QC/s03_aframr/dataset/hm3.zarr/",
#     PHENO_DIR="/u/project/sgss/PAGE/phenotype/",
# ):
#     if chrom is None:
#         chrom = np.arange(1, 23)
#     elif isinstance(chrom, List):
#         chrom = np.array(chrom)
#     elif isinstance(chrom, int):
#         chrom = np.array([chrom])
#     else:
#         raise ValueError("chrom must be None, List or int")

#     trait_cols = [
#         # Inflammtory traits
#         "crp",
#         "total_wbc_cnt",
#         "mean_corp_hgb_conc",
#         "platelet_cnt",
#         # lipid traits
#         "hdl",
#         "ldl",
#         "triglycerides",
#         "total_cholesterol",
#         # lifestyle traits
#         "cigs_per_day_excl_nonsmk_updated",
#         "coffee_cup_day",
#         # glycemic traits
#         "a1c",
#         "insulin",
#         "glucose",
#         "t2d_status",
#         # electrocardiogram traits
#         "qt_interval",
#         "qrs_interval",
#         "pr_interval",
#         # blood pressure traits
#         "systolic_bp",
#         "diastolic_bp",
#         "hypertension",
#         # anthropometric traits
#         "waist_hip_ratio",
#         "height",
#         "bmi",
#         # kidney traits
#         "egfrckdepi",
#     ]

#     covar_cols = ["study", "age", "sex", "race_ethnicity", "center"] + [
#         f"geno_EV{i}" for i in range(1, 51)
#     ]

#     race_encoding = {
#         1: "Unclassified",
#         2: "African American",
#         3: "Hispanic/Latino",
#         4: "Asian",
#         5: "Native Hawaiian",
#         6: "Native American",
#         7: "Other",
#     }

#     race_color = {
#         "Unclassified": "#ffffff",
#         "African American": "#e9d16a",
#         "Hispanic/Latino": "#9a3525",
#         "Asian": "#3c859d",
#         "Native Hawaiian": "#959f6e",
#         "Native American": "#546f7b",
#         "Other": "#d07641",
#     }

#     df_pheno = pd.read_csv(
#         join(PHENO_DIR, "MEGA_page-harmonized-phenotypes-pca-freeze2-2016-12-14.txt"),
#         sep="\t",
#         na_values=".",
#         low_memory=False,
#     )
#     df_pheno["race_ethnicity"] = df_pheno["race_ethnicity"].map(race_encoding)

#     dset_list = []
#     for i_chr in tqdm(chrom):
#         dset_list.append(xr.open_zarr(join(GENO_DIR, f"chr{i_chr}.zip")))

#     dset = xr.concat(dset_list, dim="snp")

#     df_aframr_pheno = df_pheno.set_index("PAGE_Subject_ID").loc[
#         dset.indiv.values, trait_cols + covar_cols
#     ]

#     for col in df_aframr_pheno.columns:
#         dset[f"{col}@indiv"] = ("indiv", df_aframr_pheno[col].values)

#     for col in ["center", "study", "race_ethnicity"]:
#         dset[f"{col}@indiv"] = dset[f"{col}@indiv"].astype(str)

#     # format the dataset to follow the new standards
#     for k in dset.data_vars.keys():
#         if k.endswith("@indiv"):
#             dset.coords[k.split("@")[0]] = ("indiv", dset.data_vars[k].data)
#         if k.endswith("@snp"):
#             dset.coords[k.split("@")[0]] = ("snp", dset.data_vars[k].data)
#     dset = dset.drop_vars(
#         [
#             k
#             for k in dset.data_vars.keys()
#             if (k.endswith("@indiv") or k.endswith("@snp"))
#         ]
#     )

#     dset = dset.rename({n: n.split("@")[0] for n in [k for k in dset.coords.keys()]})

#     return dset


# def load_page_eur_afr_imputed(
#     region: str,
#     imputed_vcf_dir: str = "/u/project/pasaniuc/pasaniucdata/admixture/projects/PAGE-QC/s01_dataset/all",
# ):
#     """
#     PAGE admixed individuals with EUR - AFR ancestries in imputed density

#     Parameters
#     ----------
#     region : str
#         Region name, in the format of "chrom:start-stop", e.g. "1:1-100000"
#     imputed_vcf_dir : str
#         Directory of imputed VCF files
#     """
#     chrom = int(region.split(":")[0])
#     start, stop = [int(i) for i in region.split(":")[1].split("-")]
#     print(chrom)
#     # load hm3 data set
#     dset_hm3 = load_page_eur_afr_hm3(chrom=chrom)

#     dset = admix.io.read_vcf(
#         join(imputed_vcf_dir, f"chr{chrom}.vcf.gz"),
#         region=f"{chrom}:{start}-{stop}",
#     ).sel(indiv=dset_hm3.indiv.values)

#     dset = admix.data.impute_lanc(dset=dset, dset_ref=dset_hm3)
#     dset.attrs["n_anc"] = 2
#     return dset


# UK Biobank admixed individuals with EUR - AFR ancestries in hapmap3 density
def load_ukb_eur_afr_hm3(
    chrom=None,
    GENO_DIR="/u/project/pasaniuc/pasaniucdata/admixture/projects/admix-prs-uncertainty/data/admix-analysis/dataset",
    PHENO_DIR="/u/project/pasaniuc/pasaniucdata/admixture/projects/admix-prs-uncertainty/data/pheno",
):
    if chrom is None:
        chrom = np.arange(1, 23)
    elif isinstance(chrom, List):
        chrom = np.array(chrom)
    elif isinstance(chrom, int):
        chrom = np.array([chrom])
    else:
        raise ValueError("chrom must be None, List or int")

    dset_list = []
    for i_chr in tqdm(chrom):
        dset_list.append(xr.open_zarr(join(GENO_DIR, f"chr{i_chr}.zarr")))

    dset = xr.concat(dset_list, dim="snp")

    trait_cols = ["height"]

    for trait in trait_cols:
        # raw phenotype
        df_pheno = pd.read_csv(
            join(PHENO_DIR, f"admix.{trait}.pheno"),
            delim_whitespace=True,
            low_memory=False,
        )

        df_pheno.index = df_pheno.FID.astype(str) + "_" + df_pheno.IID.astype(str)
        df_pheno = df_pheno.drop(columns=["FID", "IID"])
        dset.coords[trait] = "indiv", df_pheno[trait].reindex(dset.indiv.values)

        # residual phenotype
        df_pheno = pd.read_csv(
            join(PHENO_DIR, f"admix.{trait}.residual_pheno"),
            delim_whitespace=True,
            low_memory=False,
        )

        df_pheno.index = df_pheno.FID.astype(str) + "_" + df_pheno.IID.astype(str)
        df_pheno = df_pheno.drop(columns=["FID", "IID"])
        dset.coords[f"{trait}_residual"] = (
            "indiv",
            df_pheno[f"{trait}-residual"].reindex(dset.indiv.values),
        )

    df_covar = pd.read_csv(
        join(PHENO_DIR, f"admix.covar"),
        delim_whitespace=True,
        low_memory=False,
    )

    df_covar.index = df_covar.FID.astype(str) + "_" + df_covar.IID.astype(str)
    df_covar = df_covar.drop(columns=["FID", "IID"])

    for col in df_covar:
        dset.coords[col] = "indiv", df_covar[col].reindex(dset.indiv.values)

    dset = dset.rename({n: n.split("@")[0] for n in [k for k in dset.coords.keys()]})

    return Dataset(
        geno=np.swapaxes(dset.geno.data, 0, 1),
        lanc=np.swapaxes(dset.lanc.data, 0, 1),
        n_anc=2,
        indiv=dset.indiv.to_dataframe().drop(columns=["indiv"]),
        snp=dset.snp.to_dataframe().drop(columns=["snp"]),
    )


def load_ukb_eur_afr_imputed(
    region: str,
    imputed_vcf_dir: str = "/u/project/pasaniuc/pasaniucdata/admixture/projects/admix-prs-uncertainty/data/PLINK/admix/topmed",
):
    """
    UKB admixed individuals with EUR - AFR ancestries in imputed density

    Parameters
    ----------
    region : str
        Region name, in the format of "chrom:start-stop", e.g. "1:1-100000"
    imputed_vcf_dir : str
        Directory of imputed VCF files
    """
    chrom = int(region.split(":")[0])
    start, stop = [int(i) for i in region.split(":")[1].split("-")]
    # load hm3 data set
    dset_hm3 = load_ukb_eur_afr_hm3(chrom=chrom)

    dset = admix.io.read_vcf(
        join(imputed_vcf_dir, f"chr{chrom}/chr{chrom}.sample.imputed.vcf.gz"),
        region=f"chr{chrom}:{start}-{stop}",
    )
    if dset is None:
        warnings.warn(f"No SNPs found in {region}, `None` is returned")
        return None

    dset = dset.sel(indiv=dset_hm3.indiv.values)

    # impute local ancestry
    dset = admix.data.impute_lanc(dset=dset, dset_ref=dset_hm3)
    dset.attrs["n_anc"] = 2
    return dset


def load_lab_dataset(name: str, chrom: int = None, region: str = None) -> xr.Dataset:
    """Load prepared dataset in Bogdan lab, currently available
    if you use this function on cluster, and have the proper access to the data.
    Contact kangchenghou@gmail.com before using this function.

    Parameters
    ----------
    name : str
        Name of the dataset
        - `simulate_eur_afr.20_80`: Simulated admixture of 20% EUR and 80% AFR
        - `simulate_eur_afr.50_50`: Simulated admixture of 50% EUR and 50% AFR
        - `ukb_eur_afr_hm3`: Admixed individuals of European and African ancestries
            in UK Biobank with hapmap3 SNPs
        - `ukb_eur_afr_imputed`: Admixed individuals of European and African ancestries
            in UK Biobank with imputed SNPs
        - `page_eur_afr_hm3`: Admixed individuals of European and African ancestries
            in PAGE with hapmap3 SNPs
        - `page_eur_afr_imputed`: Admixed individuals of European and African ancestries
            in PAGE with imputed SNPs
    chrom : int, optional
        Chromosome number
    region : str, optional
        Region name

    Returns
    -------
    xr.Dataset
        dataset
    """
    if name.startswith("simulate_eur_afr"):
        assert name in [
            "simulate_eur_afr.20_80",
            "simulate_eur_afr.50_50",
        ]
        root_dir = "/u/project/pasaniuc/pasaniucdata/admixture/dataset"
        assert os.path.isdir(root_dir), f"check that you have access to {root_dir}"
        dset = xr.open_zarr(join(root_dir, name + ".zip"))
    elif name == "ukb_eur_afr_hm3":
        assert region is None, "region must not be specified when loading hm3 data"
        dset = load_ukb_eur_afr_hm3(chrom=chrom)
    elif name == "ukb_eur_afr_imputed":
        assert region is not None, "region must be specified when loading imputed data"
        assert chrom is None, "chrom must not be specified when loading imputed data"
        dset = load_ukb_eur_afr_imputed(region=region)
    else:
        raise NotImplementedError
    # elif name == "page_eur_afr_hm3":
    #     assert region is None, "region must not be specified when loading hm3 data"
    #     dset = load_page_eur_afr_hm3(chrom=chrom)
    # elif name == "page_eur_afr_imputed":
    #     assert region is not None, "region must be specified when loading imputed data"
    #     assert chrom is None, "chrom must not be specified when loading imputed data"
    #     dset = load_page_eur_afr_imputed(region=region)
    return dset
