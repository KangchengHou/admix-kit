import matplotlib.pyplot as plt
import numpy as np
from matplotlib import collections as mc
import pandas as pd


def plot_local_anc(lanc: np.ndarray):
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
    fig, ax = plt.subplots()
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
    return ax
    #   p <- ggplot(df_plot) +
    #     geom_segment(aes(
    #       x = start,
    #       y = row,
    #       xend = stop,
    #       yend = row,
    #       colour = as.factor(label),
    #     ),
    #     size = 2) +  scale_colour_brewer(palette = "Set1") + theme_classic() +
    #     labs(x = "Position",
    #          y = "Haplotype",
    #          color = "Ancestry")
    #   return(p)
    # }


# plot_local_anc <- function(local_anc) {
#   # local_anc: (n_haplo, n_snp)
#   df_plot <-
#     tibble(
#       start = numeric(),
#       stop = numeric(),
#       label = integer(),
#       row = integer()
#     )
#   n_haplo <- dim(local_anc)[1]
#   for (i_haplo in 1:n_haplo) {
#     a <- local_anc[i_haplo, ]
#     switch <- which(a[2:length(a)] != a[1:(length(a) - 1)])
#     switch <- c(1, switch, length(a))
#     first_label <- a[1]
#     for (i_switch in 1:(length(switch) - 1)) {
#       df_plot <-
#         df_plot %>% add_row(
#           start = switch[i_switch],
#           stop = switch[i_switch + 1],
#           label = a[switch[i_switch] + 1],
#           row = i_haplo
#         )
#     }
#   }
#   p <- ggplot(df_plot) +
#     geom_segment(aes(
#       x = start,
#       y = row,
#       xend = stop,
#       yend = row,
#       colour = as.factor(label),
#     ),
#     size = 2) +  scale_colour_brewer(palette = "Set1") + theme_classic() +
#     labs(x = "Position",
#          y = "Haplotype",
#          color = "Ancestry")
#   return(p)
# }