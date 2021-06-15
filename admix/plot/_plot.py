import matplotlib.pyplot as plt
import numpy as np
from matplotlib import collections as mc
import pandas as pd


def manhattan(pvals):
    pass


def lanc(lanc: np.ndarray, ax=None):
    if ax is None:
        ax = plt.gca()

    start = []
    stop = []
    label = []
    row = []
    n_haplo = lanc.shape[0]
    for i_haplo in range(n_haplo):

        a = lanc[i_haplo, :]
        switch = np.where(a[1:] != a[0:-1])[0]
        switch = np.concatenate([[0], switch, [len(a)]])

        for i_switch in range(len(switch) - 1):
            start.append(switch[i_switch])
            stop.append(switch[i_switch + 1])
            label.append(a[start[-1] + 1])
            row.append(i_haplo)

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
    ax.set_xlabel("Position")
    ax.set_ylabel("Haplotype")
    ax.set_yticks([])
    ax.set_yticklabels([])


def admixture(a, labels=None, label_orders=None, ax=None):
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
