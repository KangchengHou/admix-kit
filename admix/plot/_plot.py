import warnings
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import collections as mc
import pandas as pd
import matplotlib
import xarray as xr
import warnings


def manhattan(pvals):
    pass


def lanc(
    dset: xr.Dataset = None, lanc: np.ndarray = None, ax=None, max_indiv: int = 10
) -> None:
    """
    Plot local ancestry.

    Parameters
    ----------
    dset: xarray.Dataset
        A dataset containing the local ancestry matrix.
    lanc: np.ndarray
        A numpy array of shape (n_indiv, n_snp, 2)
    ax: matplotlib.Axes
        A matplotlib axes object to plot on. If None, will create a new one.
    max_indiv: int
        The maximum number of individuals to plot.
    Returns
    -------
    ax: matplotlib.Axes
    """
    # if dataset is provided, use it to extract lanc
    if dset is not None:
        lanc = dset.lanc.values
    else:
        assert lanc is not None, "either dataset or lanc must be provided"
    assert lanc.shape[2] == 2, "lanc must be of shape (n_indiv, n_snp, 2)"
    n_indiv, n_snp = lanc.shape[0:2]

    if n_indiv > max_indiv:
        warnings.warn(
            f"Only the first {max_indiv} are plotted. To plot more individuals, increase `max_indiv`"
        )
    else:
        max_indiv = n_indiv
    if ax is None:
        ax = plt.gca()

    start = []
    stop = []
    label = []
    row = []

    for i_indiv in range(max_indiv):
        for i_ploidy in range(2):
            a = lanc[i_indiv, :, i_ploidy]
            switch = np.where(a[1:] != a[0:-1])[0]
            switch = np.concatenate([[0], switch, [len(a)]])

            for i_switch in range(len(switch) - 1):
                start.append(switch[i_switch])
                stop.append(switch[i_switch + 1])
                label.append(a[start[-1] + 1])
                row.append(i_indiv - 0.1 + i_ploidy * 0.2)

    df_plot = pd.DataFrame({"start": start, "stop": stop, "label": label, "row": row})

    lines = [[(r.start, r.row), (r.stop, r.row)] for _, r in df_plot.iterrows()]

    cmap = plt.get_cmap("tab10")

    for i, (label, group) in enumerate(df_plot.groupby("label")):
        lc = mc.LineCollection(
            [lines[l_i] for l_i in group.index],
            linewidths=2,
            label=label,
            color=cmap(i),
        )
        ax.add_collection(lc)

    ax.legend()
    ax.autoscale()
    ax.set_xlabel("SNP index")
    ax.set_ylabel("Individuals")
    ax.set_yticks([])
    ax.set_yticklabels([])


def admixture(
    a: np.ndarray,
    labels=None,
    label_orders=None,
    ax=None,
) -> None:
    """
    Plot admixture.

    Parameters
    ----------
    a: np.ndarray
        A numpy array of shape (n_indiv, n_snp, 2)
    labels: list
        A list of labels for each individual.
    label_orders: list
        A list of orderings for the individuals.
    ax: matplotlib.Axes
        A matplotlib axes object to plot on. If None, will create a new one.

    Returns
    -------
    None
    """

    n_indiv, n_pop = a.shape

    # reorder based on labels
    if labels is not None:
        dict_label_range = dict()
        reordered_a = []
        unique_labels = np.unique(labels)

        if label_orders is not None:
            assert set(label_orders) == set(
                unique_labels
            ), "label_orders must cover all unique labels"
            unique_labels = label_orders
        cumsum = 0
        for label in unique_labels:
            reordered_a.append(a[labels == label, :])
            dict_label_range[label] = [cumsum, cumsum + sum(labels == label)]
            cumsum += sum(labels == label)
        a = np.vstack(reordered_a)

    if ax is None:
        ax = plt.gca()

    cmap = plt.get_cmap("tab10")
    bottom = np.zeros(n_indiv)

    for i_pop in range(n_pop):
        ax.bar(
            np.arange(n_indiv),
            height=a[:, i_pop],
            width=1,
            bottom=bottom,
            facecolor=cmap(i_pop),
            edgecolor=cmap(i_pop),
        )
        bottom += a[:, i_pop]

    ax.tick_params(axis="both", left=False, labelleft=False)

    if labels is not None:
        seps = sorted(np.unique(np.concatenate([r for r in dict_label_range.values()])))
        for x in seps[1:-1]:
            ax.vlines(x - 0.5, ymin=0, ymax=1, color="black")

        ax.set_xticks([np.mean(dict_label_range[label]) for label in dict_label_range])
        ax.set_xticklabels([label for label in dict_label_range])
    else:
        ax.get_xaxis().set_ticks([])
    for pos in ["top", "right", "bottom", "left"]:
        ax.spines[pos].set_visible(False)
    return ax
