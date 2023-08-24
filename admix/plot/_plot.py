import warnings
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from matplotlib import patheffects

import numpy as np
import pandas as pd
from scipy import stats

import admix
from admix.data import quantile_normalize
from admix.data import lambda_gc

from typing import Dict


def pca(
    df_pca: pd.DataFrame,
    x: str = "PC1",
    y: str = "PC2",
    label_col: str = None,
    label_order: list = None,
    s=5,
    legend_loc="on data",
    alpha=None,
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
    if alpha is None:
        alpha = 1.0
    else:
        assert isinstance(alpha, float) or isinstance(alpha, dict)
    if ax is None:
        ax = plt.gca()

    if label_order is None:
        label_order = df_pca[label_col].unique()

    ax.set_xlabel(x)
    ax.set_ylabel(y)

    if label_col is None:
        return

    # otherwise label_col is present
    for label in label_order:
        group = df_pca.loc[df_pca[label_col] == label, :]
        if isinstance(alpha, dict):
            label_alpha = alpha[label] if label in alpha else 1.0
        else:
            label_alpha = alpha
        ax.scatter(group[x], group[y], s=s, label=label, alpha=label_alpha)

    if legend_loc == "on data":

        all_pos = (
            pd.DataFrame(df_pca[[x, y, label_col]])
            .groupby(label_col, observed=True)
            .median()
            .sort_index()
        )

        for label, x_pos, y_pos in all_pos.itertuples():
            ax.text(
                x_pos,
                y_pos,
                label,
                # weight="bold",
                path_effects=[patheffects.withStroke(linewidth=2.5, foreground="w")],
                verticalalignment="center",
                horizontalalignment="center",
            )

    legend = ax.legend()
    for lh in legend.legendHandles:
        lh.set_alpha(1)
        lh.set_sizes([30])


def joint_pca(
    df_pc,
    x="PC1",
    y="PC2",
    sample_alpha=0.1,
    axes=None,
    figsize=(8.5, 4),
    label_col="SUPERPOP",
    sample_label="SAMPLE",
):
    """Joint PCA plot

    Parameters
    ----------
    df_pc : pd.DataFrame
        dataframe with PCA components
    eigenval : np.ndarray
        eigenvalues
    """
    new_axes = axes is None
    if new_axes:
        fig, axes = plt.subplots(figsize=figsize, dpi=150, ncols=2)

    admix.plot.pca(
        df_pc[df_pc[label_col] != sample_label],
        x=x,
        y=y,
        label_col=label_col,
        ax=axes[0],
    )
    assert set([x, y]).issubset(
        df_pc.columns
    ), f"{x} and {y} must be in the columns of df_pc"

    x_pos, y_pos = df_pc.columns.get_loc(x), df_pc.columns.get_loc(y)
    xlabel, ylabel = x, y
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)

    admix.plot.pca(
        df_pc,
        x=x,
        y=y,
        label_col=label_col,
        alpha={sample_label: sample_alpha},
        ax=axes[1],
    )
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(ylabel)

    if new_axes:
        return fig, axes


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
    pval = pval[~np.isnan(pval)]
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


def manhattan(
    pval,
    chrom=None,
    pos=None,
    axh_y=-np.log10(5e-8),
    s=0.1,
    label=None,
    ax=None,
    color="#3b76af",
):
    """Manhatton plot of p-values

    Parameters
    ----------
    chrom : np.ndarray
        array-like
    pval : np.ndarray
        p-values, array-like
    pos: np.ndarray
        array-like, position for each SNP, if provided, position will be used
    axh_y : np.ndarray, optional
        horizontal line for genome-wide significance, by default -np.log10(5e-8)
    s : float, optional
        dot size, by default 0.1
    ax : matplotlib.axes, optional
        axes, by default None
    """

    if ax is None:
        ax = plt.gca()

    assert (chrom is None) or (pos is None), "chrom and pos cannot be both provided"
    if pos is None:
        pos_provided = False
        pos = np.arange(len(pval))
    else:
        pos_provided = True
        assert len(pos) == len(pval)

    if chrom is None:
        # use snp index
        if pos_provided:
            ax.scatter(
                pos / 1e6,
                -np.log10(pval),
                s=s,
                label=label,
                facecolor=color,
                marker="o",
            )
            ax.set_xlabel("SNP position (Mb)")
        else:
            ax.scatter(pos, -np.log10(pval), s=s, label=label, c=color)
            ax.set_xlabel("SNP index")

    else:
        assert pos_provided is False
        assert len(chrom) == len(pval)
        color_list = ["#1b9e77", "#d95f02"]
        # plot dots for odd and even chromosomes
        for mod in range(2):
            index = np.where(chrom % 2 == mod)[0]
            ax.scatter(
                pos[index],
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


def susie(pip, dict_cs, pos=None, ax=None, cmap="Set1"):
    cmap = plt.get_cmap(cmap)
    if ax is None:
        ax = plt.gca()
    if pos is None:
        pos = np.arange(len(pip))
        pos_provided = False
    else:
        pos_provided = True
        assert len(pos) == len(pip)
        pos = pos / 1e6

    ax.scatter(x=pos, y=pip, s=3, color="gray")

    for i, cs in enumerate(dict_cs):
        cs_pos = dict_cs[cs]
        ax.scatter(
            x=pos[cs_pos],
            y=pip[cs_pos],
            s=15,
            edgecolors=cmap.colors[i],
            facecolors=cmap.colors[i],
            alpha=0.6,
        )
    if pos_provided:
        ax.set_xlabel("SNP position (Mb)")
    else:
        ax.set_xlabel("SNP index")
    ax.set_ylabel("PIP")
    ax.set_ylim(-0.05, 1.05)


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
        pos = dset.snp.POS.values
        BP_POS = True
    else:
        assert lanc is not None, "either dataset or lanc must be provided"
        pos = np.arange(lanc.shape[0])
        BP_POS = False
    # append dummy snp at the end to make plotting easier
    pos = np.concatenate([pos, [pos[-1] + 1]])

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

    for i_indiv in range(n_plot_indiv):
        for i_ploidy in range(2):
            a = lanc[:, i_indiv, i_ploidy]
            switch = np.where(a[1:] != a[0:-1])[0]
            switch = np.concatenate([[0], switch, [len(a)]])
            for i_switch in range(len(switch) - 1):
                start_idx, stop_idx = switch[i_switch], switch[i_switch + 1]
                if BP_POS:
                    start.append(pos[start_idx] / 1e6)
                    stop.append(pos[stop_idx] / 1e6)
                else:
                    start.append(start_idx)
                    stop.append(stop_idx)

                label.append(a[start_idx + 1])
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

    if BP_POS:
        ax.set_xlabel("SNP position (Mb)")
    else:
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


def compare_pval(
    x_pval: np.ndarray,
    y_pval: np.ndarray,
    xlabel: str = None,
    ylabel: str = None,
    ax=None,
    s: int = 5,
):
    """Compare two p-values.

    Parameters
    ----------
    x_pval: np.ndarray
        The p-value for the first variable.
    y_pval: np.ndarray
        The p-value for the second variable.
    xlabel: str
        The label for the first variable.
    ylabel: str
        The label for the second variable.
    ax: matplotlib.Axes
        A matplotlib axes object to plot on. If None, will create a new one.
    """
    if ax is None:
        ax = plt.gca()
    if not isinstance(x_pval, np.ndarray):
        x_pval = np.array(x_pval)
    if not isinstance(y_pval, np.ndarray):
        y_pval = np.array(y_pval)
    nonnan_idx = ~np.isnan(x_pval) & ~np.isnan(y_pval)
    x_pval, y_pval = -np.log10(x_pval[nonnan_idx]), -np.log10(y_pval[nonnan_idx])
    ax.scatter(x_pval, y_pval, s=s)
    lim = max(np.nanmax(x_pval), np.nanmax(y_pval)) * 1.1
    ax.plot([0, lim], [0, lim], "k--", alpha=0.5, lw=1, label="y=x")

    # add a regression line
    slope = np.linalg.lstsq(x_pval[:, None], y_pval[:, None], rcond=None)[0].item()

    ax.axline(
        (0, 0),
        slope=slope,
        color="black",
        ls="--",
        lw=1,
        label=f"y={slope:.2f} x",
    )
    ax.legend()
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)


def rg_posterior(
    xs: np.ndarray,
    dict_loglik: Dict[str, np.ndarray],
    ci=[0.5, 0.95],
    s=11,
    colors="black",
    markers="o",
    ax=None,
):
    """
    Plot the posterior distribution

    Parameters
    ----------
    xs: np.ndarray
        list of x coordinates
    dict_loglik: Dict[np.ndarray]
        trait -> list of log-likelihoods
    ci: Union[float, List[float]]
        ci to plot, can be 1 float or two float
    colors:
        ["darkblue"] * (len(est) - 1) + ["darkred"]
    markers:
        ["o"] * (len(est) - 1) + ["^"]
    """
    if ax is None:
        ax = plt.gca()

    assert len(ci) == 2, "Currently must plot 2 CIs"
    assert ci[0] < ci[1], "Smaller CI should come first"
    assert np.all([len(xs) == len(dict_loglik[t]) for t in dict_loglik])

    trait_list = list(dict_loglik.keys())[::-1]
    if isinstance(colors, list):
        colors = colors[::-1]
    elif isinstance(colors, str):
        colors = [colors] * len(trait_list)
    if isinstance(markers, list):
        markers = markers[::-1]
    elif isinstance(markers, str):
        markers = [markers] * len(trait_list)

    dict_mode = {trait: xs[dict_loglik[trait].argmax()] for trait in trait_list}

    dict_ci_err: Dict[int, Dict] = {ci[0]: dict(), ci[1]: dict()}

    for trait in trait_list:
        mode = dict_mode[trait]
        for each_ci in ci:
            hdi = admix.data.hdi(xs, dict_loglik[trait], ci=each_ci)
            assert not isinstance(
                hdi, list
            ), f"HPDI for {trait} contains multiple intervals {hdi}, indicating lack of data. Please rerun this function after remove this trait."
            dict_ci_err[each_ci][trait] = [mode - hdi[0], hdi[1] - mode]

    mode = np.array([dict_mode[trait] for trait in trait_list])
    lw_list = [2.5, 1.0]
    for i, each_ci in enumerate(ci):
        ci_low = [dict_ci_err[each_ci][trait][0] for trait in trait_list]
        ci_high = [dict_ci_err[each_ci][trait][1] for trait in trait_list]
        ax.errorbar(
            y=np.arange(len(mode)),
            x=mode,
            xerr=(ci_low, ci_high),
            fmt=" ",
            lw=lw_list[i],
            ecolor=colors,
        )
        if i == 0:
            for j, trait in enumerate(trait_list):
                ax.scatter(
                    x=mode[j],
                    y=j,
                    marker=markers[j],
                    color=colors[j],
                    s=s,
                )

    for y in np.arange(len(mode)):
        ax.axhline(y=y, color="gray", ls="dotted", lw=0.5, alpha=0.8)

    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Highest probability density of $r_{admix}$")
    ax.set_yticks(np.arange(len(mode)))
    ax.set_ylim(-0.5, len(mode) - 0.5)
    ax.set_yticklabels(
        trait_list,
        fontsize=8,
    )

    # annotation
    ax.tick_params(left=False, pad=-1)
    ax.axvline(x=1.0, color="red", ls="--", lw=0.8, alpha=0.4)
    ax.set_title("Estimated $r_{admix}$", fontsize=10, x=0.5)
