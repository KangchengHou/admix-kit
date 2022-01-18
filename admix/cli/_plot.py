import admix
import pandas as pd
import dapgen
import numpy as np
from typing import List
from ._utils import log_params


# def merge_dataset(path_list: str, out: str):
#     """Merge multiple dataset [in zarr format] into one dataset, assuming the individiduals
#     are shared typically used for merging multiple datasets from different chromosomes.

#     Parameters
#     ----------
#     path_list : List[str]
#         path of a text file pointing to the list of paths
#     out : str
#         Path to the output zarr file
#     """
#     import xarray as xr

#     dset_list = [xr.open_zarr(p) for p in path_list]

#     dset = xr.concat(dset_list, dim="snp")

#     dset = dset.chunk(chunks={"indiv": -1, "ploidy": -1, "snp": "auto"}).compute()
#     dset.to_zarr(out, mode="w", safe_chunks=False)


def plot_pca(
    pfile: str,
    pca: str,
    out: str,
    label_col: str = None,
    x: str = "PC1",
    y: str = "PC2",
):
    """Plot PCA results to a file

    Parameters
    ----------
    pfile : str
        pfile
    label_col : str
        column in .psam file
    pca : str
        path to the pca file
    out : str
        path to the output file
    x : str
        x-axis (default PC1)
    y : str
        y-axis (default PC2)
    """
    log_params("plot-pca", locals())

    import matplotlib.pyplot as plt
    import matplotlib as mpl

    mpl.style.use("classic")

    df_psam = dapgen.read_psam(pfile + ".psam")
    df_pca = pd.read_csv(pca, delim_whitespace=True, index_col=0)
    assert np.all(df_psam.index == df_pca.index)

    df_plot = pd.merge(df_psam, df_pca, left_index=True, right_index=True)

    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    admix.plot.pca(df_plot, x=x, y=y, label_col=label_col)

    # make xticklabels and yticklabels smaller
    ax.tick_params(axis="x", labelsize=6)
    ax.tick_params(axis="y", labelsize=6)
    # make legend font smaller
    ax.legend(fontsize=8)
    plt.savefig(out, bbox_inches="tight", dpi=300)