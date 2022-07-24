import admix
import pandas as pd
import numpy as np
from typing import List
from ._utils import log_params


# def plot_pca(
#     pfile: str,
#     pca: str,
#     out: str,
#     label_col: str = None,
#     x: str = "PC1",
#     y: str = "PC2",
# ):
#     """Plot PCA results to a file

#     Parameters
#     ----------
#     pfile : str
#         pfile
#     label_col : str
#         column in .psam file
#     pca : str
#         path to the pca file
#     out : str
#         path to the output file
#     x : str
#         x-axis (default PC1)
#     y : str
#         y-axis (default PC2)
#     """
#     log_params("plot-pca", locals())

#     import matplotlib.pyplot as plt
#     import matplotlib as mpl

#     mpl.style.use("classic")

#     df_psam = dapgen.read_psam(pfile + ".psam")
#     df_pca = pd.read_csv(pca, delim_whitespace=True, index_col=0)
#     assert np.all(df_psam.index == df_pca.index)

#     df_plot = pd.merge(df_psam, df_pca, left_index=True, right_index=True)

#     fig, ax = plt.subplots(figsize=(2.5, 2.5))
#     admix.plot.pca(df_plot, x=x, y=y, label_col=label_col)

#     # make xticklabels and yticklabels smaller
#     ax.tick_params(axis="x", labelsize=6)
#     ax.tick_params(axis="y", labelsize=6)
#     # make legend font smaller
#     ax.legend(fontsize=8)
#     plt.savefig(out, bbox_inches="tight", dpi=300)


def plot_joint_pca(
    ref_pfile: str,
    pca_prefix: str,
    out: str,
    sample_alpha: float = 0.1,
    figsize=(8.5, 4),
    x="PC1",
    y="PC2",
):
    """Plot individuals on a joint PC plot assuming a joint pca of
    (1) reference dataset (2) sample dataset
    has been performed.

    Parameters
    ----------
    ref_pfile : str
        reference panel pfile prefix
    pca_prefix : str
        joint pca results prefix. {pca_prefix}.eigenvec, {pca_prefix}.eigenval
        will be read
    out : str
        output prefix. {out}.XX, {out}.YY will be produced
    sample_alpha : float
        transparency for sample dots
    figsize : tuple
        figure size (width, height), default (8, 4)
    x : str
        x-axis (default PC1)
    y : str
        y-axis (default PC2)
    """
    log_params("plot-joint-pca", locals())
    import matplotlib.pyplot as plt

    df_plot, eigenval = admix.io.read_joint_pca(
        ref_pfile=ref_pfile, pca_prefix=pca_prefix
    )
    assert set([x, y]).issubset(
        df_plot.columns
    ), f"{x} and {y} must be in the columns of {pca_prefix}.eigenvec file"

    fig, axes = plt.subplots(figsize=figsize, dpi=150, ncols=2)
    admix.plot.joint_pca(
        df_pc=df_plot,
        eigenval=eigenval,
        axes=axes,
        sample_alpha=sample_alpha,
        x=x,
        y=y,
    )
    fig.tight_layout()
    fig.savefig(f"{out}.png", bbox_inches="tight")
    admix.logger.info(f"PCA plots saved to {out}.png")
