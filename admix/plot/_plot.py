import warnings
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import collections as mc
import pandas as pd
import warnings
from scipy import stats
from admix.data import quantile_normalize
from admix.data import lambda_gc
import seaborn as sns
import admix


def pca(
    df_pca: pd.DataFrame,
    x: str = "PC1",
    y: str = "PC2",
    label_col: str = None,
    s=5,
    ax=None,
):
    """PCA plot

    Parameters
    ----------
    df_pca : pd.DataFrame
        dataframe with PCA components
    x : str, optional
        x-axis, by default "PC1"
    y : str, optional
        y-axis, by default "PC2"
    label_col : str, optional
        column name for labels, by default None
    s : float, optional
    """
    if ax is None:
        ax = plt.gca()
    sns.scatterplot(data=df_pca, x=x, y=y, hue=label_col, linewidth=0, s=s, ax=ax)


def qq(pval, label=None, ax=None, bootstrap_ci=False):
    """qq plot of p-values

    Parameters
    ----------
    pval : np.ndarray
        p-values, array-like
    ax : matplotlib.axes, optional
        by default None
    return_lambda_gc : bool, optional
        whether to return the lambda GC, by default False
    """
    if ax is None:
        ax = plt.gca()

    pval = np.array(pval)
    expected_pval = stats.norm.sf(quantile_normalize(-pval))
    ax.scatter(-np.log10(expected_pval), -np.log10(pval), s=2, label=label)
    lim = max(-np.log10(expected_pval))
    ax.plot([0, lim], [0, lim], "r--")
    ax.set_xlabel("Expected -$\log_{10}(p)$")
    ax.set_ylabel("Observed -$\log_{10}(p)$")
    if bootstrap_ci == True:
        lgc, lgc_ci = lambda_gc(pval, bootstrap_ci=True)
    else:
        lgc = lambda_gc(pval, bootstrap_ci=False)

    if bootstrap_ci:
        print(f"lambda GC: {lgc:.3g} [{lgc_ci[0]:.3g}, {lgc_ci[1]:.3g}]")
        return lgc, lgc_ci
    else:
        print(f"lambda GC: {lgc:.3g}")
        return lgc


def manhattan(pval, chrom=None, axh_y=-np.log10(5e-8), s=0.1, label=None, ax=None):
    """Manhatton plot of p-values

    Parameters
    ----------
    chrom : np.ndarray
        array-like
    pval : np.ndarray
        p-values, array-like
    axh_y : np.ndarray, optional
        horizontal line for genome-wide significance, by default -np.log10(5e-8)
    s : float, optional
        dot size, by default 0.1
    ax : matplotlib.axes, optional
        axes, by default None
    """
    if ax is None:
        ax = plt.gca()

    if chrom is None:
        # use snp index
        ax.scatter(np.arange(len(pval)), -np.log10(pval), s=s, label=label)
        ax.set_xlabel("SNP index")

    else:
        assert len(chrom) == len(pval)
        color_list = ["#1b9e77", "#d95f02"]
        # plot dots for odd and even chromosomes
        for mod in range(2):
            index = np.where(chrom % 2 == mod)[0]
            ax.scatter(
                np.arange(len(pval))[index],
                -np.log10(pval)[index],
                s=s,
                color=color_list[mod],
                label=label,
            )

        # label unique chromosomes
        xticks = []
        xticklabels = []
        for chrom_i in np.unique(chrom):
            xticks.append(np.where(chrom == chrom_i)[0].mean())
            xticklabels.append(chrom_i)

        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, rotation=90, fontsize=8)
        ax.set_xlabel("Chromosome")

    ax.set_ylabel("-$\log_{10}(P)$")
    if axh_y is not None:
        ax.axhline(y=axh_y, color="r", ls="--")


def lanc(
    dset: admix.Dataset = None,
    lanc: np.ndarray = None,
    ax=None,
    max_indiv: int = None,
) -> None:
    """
    Plot local ancestry.

    Parameters
    ----------
    dset: xarray.Dataset
        A dataset containing the local ancestry matrix.
    lanc: np.ndarray
        A numpy array of shape (n_snp, n_indiv, 2)
    ax: matplotlib.Axes
        A matplotlib axes object to plot on. If None, will create a new one.
    max_indiv: int
        The maximum number of individuals to plot.
        If None, will plot the first 10 individuals
    Returns
    -------
    ax: matplotlib.Axes
    """
    # if dataset is provided, use it to extract lanc
    if dset is not None:
        lanc = dset.lanc.compute()
    else:
        assert lanc is not None, "either dataset or lanc must be provided"
    assert lanc.shape[2] == 2, "lanc must be of shape (n_snp, n_indiv, 2)"
    n_snp, n_indiv = lanc.shape[0:2]

    if max_indiv is not None:
        n_plot_indiv = min(max_indiv, n_indiv)
    else:
        n_plot_indiv = min(n_indiv, 10)
        if n_plot_indiv < n_indiv:
            warnings.warn(
                f"Only the first {n_plot_indiv} are plotted. To plot more individuals, increase `max_indiv`"
            )
    if ax is None:
        ax = plt.gca()

    start = []
    stop = []
    label = []
    row = []

    # TODO: extend the label categories such that n_anc labels in df_plot
    for i_indiv in range(n_plot_indiv):
        for i_ploidy in range(2):
            a = lanc[:, i_indiv, i_ploidy]
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


def compare_pval(x_pval, y_pval, xlabel=None, ylabel=None, ax=None, s=5):
    if ax is None:
        ax = plt.gca()
    ax.scatter(-np.log10(x_pval), -np.log10(y_pval), s=s)
    lim = max(np.nanmax(-np.log10(x_pval)), np.nanmax(-np.log10(y_pval))) * 1.1
    ax.plot([0, lim], [0, lim], "k--", alpha=0.5, lw=1)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
