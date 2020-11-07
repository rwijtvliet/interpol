# %% Illustration of interpolation function in/around single polygon.

import interpol as ip
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle("Interpolation in/around polygon, with `polygon` function.")
pix = 150
count = 7

anchors = np.array(
    [(np.cos(a), np.sin(a)) for a in np.pi * np.linspace(0.25, 2.25, count, False)]
) + 0.5 * np.random.rand(count, 2)
extremes = np.array([anchors.min(axis=0), anchors.max(axis=0)])
extremes = np.array([[1.3, -0.3], [-0.3, 1.3]]) @ extremes  # add 30% margin
extent = extremes.T.flatten()  # xmin xmax ymin ymax

for j, ax in enumerate(axes):
    if j == 0:
        values = np.random.rand(count) + 0.1
        ax.set_title("Anchor points define floats")
    else:
        values = np.random.rand(count, 3)
        ax.set_title("Anchor points define colors")

    f = ip.polygon(anchors, values)
    data = np.array(
        [
            [f((x, y)) for x in np.linspace(extent[0], extent[1], pix)]
            for y in np.linspace(extent[2], extent[3], pix)
        ]
    )
    ax.imshow(
        data, origin="lower", extent=extent, cmap="gist_heat", vmin=0, vmax=data.max()
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.plot(*anchors.T, "ko", markersize=4)
    ax.plot(*anchors.T, "wo", markersize=2)
    ax.plot(*np.array([*anchors, anchors[0]]).T, "gray", linewidth=0.5)

fig.tight_layout(rect=[0, 0.05, 1, 0.95])
