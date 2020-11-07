# %% Illustration of interpolation functions in/around set of points.

import interpol as ip
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(3, 2, figsize=(10, 15))
fig.suptitle(
    "Interpolation around set of points with `triangles` and `polygons` functions."
)
pix = 100
count = 10

anchors = np.random.rand(count, 2)
extremes = np.array([anchors.min(), anchors.max()])  # ensure squareness
extremes = np.array([[1.3, -0.3], [-0.3, 1.3]]) @ extremes  # add 30% margin


for i in range(2):
    if i == 0:
        values = np.random.rand(count)
        data = np.zeros((pix, pix, 2), float)
    else:
        values = np.random.rand(count, 3)
        data = np.zeros((pix, pix, 2, 3), float)

    f0 = ip.triangles(anchors, values)
    f1 = ip.polygons(anchors, values)

    data = np.array(
        [
            [[f0((x, y)), f1((x, y))] for x in np.linspace(*extremes, pix)]
            for y in np.linspace(*extremes, pix)
        ]
    )

    for j in range(2):
        axes[i + 1, j].imshow(
            data[:, :, j],
            origin="lower",
            extent=list(extremes) * 2,
            cmap="gist_heat",
            vmin=data.min(),
            vmax=data.max(),
        )

for j, shapes in enumerate([f0.delaunay.simplices, f1.polygonate.shapes]):
    for shape in shapes:
        for vi in zip(shape, np.roll(shape, 1)):
            axes[0, j].plot(*anchors[vi, :].T, "gray", linewidth=0.5)

for i in range(3):
    for j in range(2):
        ax = axes[i, j]
        ax.set_xlim(extremes)
        ax.set_ylim(extremes)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.plot(*anchors.T, "ko", markersize=4)
        ax.plot(*anchors.T, "wo", markersize=2)
        title = [
            "Tesselation...",
            "Interpolation with floats...",
            "Interpolatation with colors...",
        ][i] + "\n"
        title += ["...using `triangles`", "...using `polygons`", "delta"][j]
        ax.set_title(title)

fig.tight_layout(rect=[0, 0.05, 1, 0.95])
