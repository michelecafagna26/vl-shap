import matplotlib.pyplot as plt
import numpy as np


def plot_masks(masks, subplot_size=None, labels=None):

    if subplot_size:
        (r, c) = subplot_size
    else:
        r = np.sqrt(len(masks)).astype(int)
        c = int(len(masks)/r)+1
    figure, axis = plt.subplots(r, c)

    for ii in range(r):
        for jj in range(c):

            item_idx = ii*c+jj

            if item_idx < len(masks):
                m = masks[item_idx]

                if labels:
                    l = labels[item_idx]
                else:
                    l = item_idx

                axis[ii, jj].matshow(m)
                axis[ii, jj].set_title(f"{l}")
                axis[ii, jj].tick_params(left=False, right=False, labelleft=False,
                                         labelbottom=False, bottom=False, top=False, labeltop=False)
            else:

                # empty plot
                axis[ii, jj].set_box_aspect(m.shape[0] / m.shape[1])
                axis[ii, jj].tick_params(left=False, right=False, labelleft=False, labelbottom=False,
                                         bottom=False, top=False, labeltop=False)
    figure.tight_layout()

    return figure, axis


def compute_contribution_map(maps, shap_values):
    contrib_map = np.zeros(maps[0].shape)
    for i, m in enumerate(maps):
        contrib_map += m.astype(np.single) * shap_values[i]

    return contrib_map


def heatmap(img, maps, shap_values, vmin=None, vmax=None, alpha=0.5):

    contrib_map = compute_contribution_map(maps, shap_values)

    fig, ax = plt.subplots()

    # assuming image has the same size of the mask
    ax.imshow(img, zorder=0)
    im = ax.imshow(contrib_map, cmap="coolwarm_r", interpolation="nearest", vmin=vmin, vmax=vmax, zorder=1, alpha=alpha)
    bar = plt.colorbar(im)
    ax.set_xticks([], [])
    ax.set_yticks([], [])

    return fig, ax, im, bar


def barh(labels, shap_values, reverse=False):

    kv = {k: v for k, v in zip(labels, shap_values)}
    kv = dict(sorted(kv.items(), key=lambda x: x[1], reverse=reverse))

    fig, ax = plt.subplots()
    colors = [(0.90, 0.17, 0.31) if v < 0.0 else (0.49, 0.73, 0.91) for v in kv.values()]
    ax.barh(list(kv.keys()), list(kv.values()), color=colors)

    return fig, ax