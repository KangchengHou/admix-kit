"""
This file is not used anywhere in current version of the code.
"""

import numpy as np
import statsmodels.api as sm
from scipy import stats


def linear_f_test1(var, cov, pheno, var_size, test_vars):
    n_indiv = var.shape[0]
    n_var = var.shape[1]
    n_cov = cov.shape[1]

    design = np.zeros((n_indiv, var_size + n_cov))
    design[:, var_size : var_size + n_cov] = cov

    n_test = int(n_var / var_size)
    pvalues = np.zeros(n_test)

    for i_test in range(n_test):
        design[:, 0:var_size] = var[:, i_test * var_size : (i_test + 1) * var_size]
        beta = np.linalg.lstsq(design, pheno, rcond=None)[0]
        sigma = np.square(pheno - np.dot(design, beta)).sum() / (
            n_indiv - var_size - n_cov
        )
        iXtX = np.linalg.inv(np.dot(design.T, design))
        f_stat = (
            beta[test_vars]
            .T.dot(np.linalg.inv(iXtX[np.ix_(test_vars, test_vars)]))
            .dot(beta[test_vars])
        ) / (len(test_vars) * sigma)
        pvalues[i_test] = stats.f.sf(f_stat, len(test_vars), n_indiv - var_size - n_cov)

    return pvalues


def linear_f_test2(var, cov, pheno, var_size, test_vars):
    n_indiv = var.shape[0]
    n_var = var.shape[1]
    n_cov = cov.shape[1]

    design = np.zeros((n_indiv, var_size + n_cov))
    design[:, var_size : var_size + n_cov] = cov

    n_test = int(n_var / var_size)
    pvalues = np.zeros(n_test)

    reduced_index = np.concatenate(
        [
            [i for i in range(var_size) if i not in test_vars],
            np.arange(var_size, var_size + n_cov),
        ]
    ).astype(int)

    for i_test in range(n_test):
        design[:, 0:var_size] = var[:, i_test * var_size : (i_test + 1) * var_size]

        beta1 = np.linalg.lstsq(design[:, reduced_index], pheno, rcond=None)[0]
        ss1 = np.square(pheno - np.dot(design[:, reduced_index], beta1)).sum()
        p1 = len(beta1)
        beta2 = np.linalg.lstsq(design, pheno, rcond=None)[0]
        ss2 = np.square(pheno - np.dot(design, beta2)).sum()
        p2 = len(beta2)

        fstat = ((ss1 - ss2) / (p2 - p1)) / (ss2 / (n_indiv - p2))
        pvalues[i_test] = stats.f.sf(fstat, p2 - p1, n_indiv - p2)

    return pvalues


def linear_lrt(var, cov, pheno, var_size, test_vars):
    n_indiv = var.shape[0]
    n_var = var.shape[1]
    n_cov = cov.shape[1]

    design = np.zeros((n_indiv, var_size + n_cov))
    design[:, var_size : var_size + n_cov] = cov

    n_test = int(n_var / var_size)
    pvalues = np.zeros(n_test)

    reduced_index = np.concatenate(
        [
            [i for i in range(var_size) if i not in test_vars],
            np.arange(var_size, var_size + n_cov),
        ]
    ).astype(int)

    for i_test in range(n_test):
        design[:, 0:var_size] = var[:, i_test * var_size : (i_test + 1) * var_size]
        llf1 = sm.OLS(pheno, design[:, reduced_index]).fit().llf
        llf2 = sm.OLS(pheno, design).fit().llf
        pvalues[i_test] = stats.chi2.sf(-2 * (llf1 - llf2), len(test_vars))
    return pvalues


def linear_f_test3(var, cov, pheno, var_size, test_vars):
    n_indiv = var.shape[0]
    n_var = var.shape[1]
    n_cov = cov.shape[1]

    design = np.zeros((n_indiv, var_size + n_cov))
    design[:, var_size : var_size + n_cov] = cov

    n_test = int(n_var / var_size)
    pvalues = np.zeros(n_test)

    f_test_r_matrix = np.zeros((len(test_vars), design.shape[1]))
    for i, v in enumerate(test_vars):
        f_test_r_matrix[i, v] = 1

    for i_test in range(n_test):
        design[:, 0:var_size] = var[:, i_test * var_size : (i_test + 1) * var_size]
        model = sm.OLS(pheno, design, missing="drop").fit()
        print(model.f_test(f_test_r_matrix))
        pvalues[i_test] = model.f_test(f_test_r_matrix).pvalue.item()
    return pvalues