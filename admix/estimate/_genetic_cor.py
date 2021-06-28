import numpy as np
from scipy import linalg
import pandas as pd


def trace_mul(a, b):
    """
    Trace of two matrix inner product
    """
    assert np.all(a.shape == b.shape)
    return np.sum(a.flatten() * b.flatten())


def gen_cor(K1, K2, y):
    # TODO: check y shape
    n_indiv, n_sim = y.shape
    quad_form_func = lambda x, A: np.dot(np.dot(x.T, A), x)

    design = np.array(
        [
            [trace_mul(K1, K1), trace_mul(K1, K2), np.trace(K1)],
            [trace_mul(K2, K1), trace_mul(K2, K2), np.trace(K2)],
            [np.trace(K1), np.trace(K2), n_indiv],
        ]
    )
    df_rls = []
    for i_sim in range(n_sim):
        y_sim = y[:, i_sim]
        response = np.array(
            [
                quad_form_func(y_sim, K1),
                quad_form_func(y_sim, K2),
                np.sum(y_sim * y_sim),
            ]
        )
        rls = linalg.solve(design, response)
        df_rls.append(rls)

    df_rls = np.vstack(df_rls)
    df_rls = pd.DataFrame(df_rls, columns=["var_g", "gamma", "var_e"])
    # print("Mean:")
    # print(df_rls.mean(axis=0))
    # print("SD:")
    # print(df_rls.std(axis=0))
    return df_rls