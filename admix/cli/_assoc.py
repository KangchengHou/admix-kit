import admix
import pandas as pd
from typing import Union, List
from admix import logger
from ._utils import log_params


def assoc(
    pfile: str,
    pheno: str,
    pheno_col: str,
    out: str,
    covar: str = None,
    method: Union[str, List[str]] = "ATT",
    family: str = "quant",
    pheno_quantile_normalize: bool = False,
    covar_quantile_normalize: bool = False,
    fast: bool = True,
):
    """
    Perform association testing.

    Parameters
    ----------
    pfile : str
        Prefix to the PLINK2 file (.pgen should not be added). If method that requires
        local ancestry is specified, a matched :code:`<pfile>.lanc` file should exist.
    pheno : str
        Path to the phenotype file. The text file should be space delimited with header 
        and one individual per row. The first column should be the individual ID. Use 
        :code:`--pheno-col` to specify the column for the phenotype value 
    pheno_col : str
        Column name for the phenotype value. NaN should be encoded as "NA" and these 
        individuals will be removed in the analysis. Binary phenotype should be encoded 
        as 0 and 1, and :code:`--family binary` should be used.
    covar: str
        Path to the covariate file. The text file should be space delimited with header 
        and one individual per row. The first column should be the individual ID, and 
        the remaining columns should be the covariate values. All columns will be used
        for the analysis. NaN should be encoded as "NA" and NaN will be imputed with 
        the mean of each covariate. Categorical covariates will be converted to one
        hot encodings by the program.
    out : str
        Path the output file. <out>.<method>.assoc will be created. 
    method : Union[str, List[str]]
        Method to use for association analysis (default ATT). Other methods include:
        TRACTOR, ADM, SNP1. 
    family : str
        Family to use for association analysis (default quant). One of :code:`quant` or 
        :code:`binary`.
    pheno_quantile_normalize : bool
        Whether to quantile normalize the phenotype.
    covar_quantile_normalize : bool
        Whether to quantile normalize each column of the covariates.
    fast : bool
        Whether to use fast mode (default True).

    Examples
    --------
    .. code-block:: bash

        # See complete example at 
        # https://kangchenghou.github.io/admix-kit/quickstart-cli.html
        admix assoc \\
            --pfile toy-admix \\
            --pheno toy-admix.pheno \\
            --pheno-col SIM0 \\
            --covar toy-admix.covar \\
            --method ATT,TRACTOR \\
            --pheno-quantile-normalize True \\
            --covar-quantile-normalize True \\
            --out toy-admix
    """
    log_params("assoc", locals())
    assert family in ["quant", "binary"], "family must be either quant or binary"

    # TODO: infer block size using memory use in dask-pgen read
    dset = admix.io.read_dataset(pfile)
    admix.logger.info(f"{dset.n_snp} SNPs and {dset.n_indiv} individuals are loaded")

    df_pheno = pd.read_csv(pheno, delim_whitespace=True, index_col=0, low_memory=False)[
        [pheno_col]
    ]
    dset.append_indiv_info(df_pheno, force_update=True)
    # adding covariates
    if covar is not None:
        df_covar = pd.read_csv(
            covar, delim_whitespace=True, index_col=0, low_memory=False
        )
        dset.append_indiv_info(df_covar, force_update=True)
        covar_cols = df_covar.columns
        admix.logger.info(
            f"{len(covar_cols)} covariates are loaded: {','.join(covar_cols)}"
        )
    else:
        covar_cols = None
        admix.logger.info("No covariates are loaded")
    # retain only individuals with non-missing phenotype,
    # or with non-completely missing covariate
    indiv_mask = ~dset.indiv[pheno_col].isna().values
    if covar_cols is not None:
        covar_mask = ~(dset.indiv[covar_cols].isna().values.all(axis=1))
        indiv_mask &= covar_mask
    dset = dset[:, indiv_mask]
    admix.logger.info(
        f"{dset.n_snp} SNPs and {dset.n_indiv} individuals left "
        "after filtering for missing phenotype, or completely missing covariate"
    )

    if isinstance(method, str):
        method = [method]

    # extract pheno_values and covar_values
    pheno_values = dset.indiv[pheno_col].values
    if covar_cols is not None:
        covar_values = admix.data.convert_dummy(dset.indiv[covar_cols]).values
    else:
        covar_values = None

    if pheno_quantile_normalize:
        pheno_values = admix.data.quantile_normalize(pheno_values)
    if covar_quantile_normalize:
        for i in range(covar_values.shape[1]):
            covar_values[:, i] = admix.data.quantile_normalize(covar_values[:, i])

    dict_rls = {}
    for m in method:
        admix.logger.info(f"Performing association analysis with method {m}")
        dict_rls[m] = admix.assoc.marginal(
            dset=dset,
            pheno=pheno_values,
            cov=covar_values,
            method=m,
            family="logistic" if family == "binary" else "linear",
            fast=fast,
        )
    for m in dict_rls:
        dict_rls[m].to_csv(
            f"{out}.{m}.assoc", sep="\t", float_format="%.6g", na_rep="NA"
        )
        logger.info(f"Output written to {out}.{m}.assoc")