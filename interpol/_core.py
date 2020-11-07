"""
Interpolation functions.

2020-09 rwijtvliet@gmail.com
"""

import numpy as np
from typing import Iterable, Callable
from scipy.spatial import Delaunay, ConvexHull
from polygonation import Polygonate


#%% Interpolation function in single polygon.


def polygon(anchorpoints: Iterable, anchorvalues: Iterable) -> Callable:
    """
    Generalised barycentric interpolation inside/near single polygon.

    Interpolate the value at any point in the plane, if the coordinates and the
    values of a polygon's nodes are specified.

    Parameters
    ----------
    anchorpoints
        Iterable of node coordinates (x, y) in order as they appear in the polygon.
    anchorvalues
        Iterable of floats specifying the known value at each node.

    Returns
    -------
    Function that accepts coordinates (x, y) of any point in the plane as a
    2-Iterable, and returns the interpolated value at that point.

    Notes
    -----
    Instead of single floats, the ``anchorvalues`` may also be iterables of uniform
    length, e.g. 3- or 4-tuples specifying colors. The interpolation function's
    return value has the same length as the anchorvalues.

    For points that lie outside the polygon, the interpolated value may be smaller
    (larger) than the minimum (maximum) of the specified ``anchorvalues``. This
    means that, when using colors for the ``anchorvalues``, an invalid color may
    be returned.

    Source of algorithm: https://www.inf.usi.ch/hormann/papers/Hormann.2006.MVC.pdf
    """
    anchorpoints = np.array(anchorpoints)
    anchorvalues = np.array(anchorvalues)
    if len(anchorpoints) != len(anchorvalues):
        raise ValueError(
            "Parameters 'anchorpoints' and 'anchorvalues' must be of equal length."
        )
    if len(anchorpoints) < 3:
        raise ValueError("At least 3 anchorpoints must be specified.")

    F = anchorvalues
    F_next = np.roll(anchorvalues, -1, axis=0)
    eps = 1e-7

    def interp(point: Iterable):
        S = anchorpoints - point  # distance vector to each anchorpoint
        R = np.linalg.norm(S, axis=1)  # distances to each anchorpoint
        for r, f in zip(R, F):
            if -eps < r < eps:
                return f  # point is on anchor point

        S_next = np.roll(S, -1, axis=0)  # neighbors
        R_next = np.roll(R, -1)
        A = np.array(
            [np.linalg.det([s, s_next]) for s, s_next in zip(S, S_next)]
        )  # determinant of each displ. vector with neighbor
        D = np.array([np.dot(s, s_next) for s, s_next in zip(S, S_next)])
        for a, d, r, r_next, f, f_next in zip(A, D, R, R_next, F, F_next):
            if -eps < a < eps and d < 0:
                return (r_next * f + r * f_next) / (
                    r + r_next
                )  # point is on edge between anchor points

        T = np.array([a / (r * r_next + d) for a, d, r, r_next in zip(A, D, R, R_next)])
        T_prev = np.roll(T, 1)
        W = np.array([(t_prev + t) / r for t_prev, t, r in zip(T_prev, T, R)])
        return W.dot(F) / W.sum()

    return interp


#%% Interpolation functions in plane with points.


def triangles(
    anchorpoints: Iterable, anchorvalues: Iterable, outsidevalue=None
) -> Callable:
    """
    Standard barycentric interpolation (using triangles) near set of points.

    Interpolate the value at any point in the plane, if the coordinates and
    values of a set of anchors is specified. If >3 anchors are specified, the
    convex hull around them is tessellated with Delaunay triangles.

    Parameters
    ----------
    anchorpoints
        Iterable of anchor point coordinates (x, y).
    anchorvalues
        Iterable of floats specifying the known value at each anchor point.
    outsidevalue, optional
        Float specifying the value for points outside the hull. If not specified,
        an attempt at extrapolation is made (default).

    Returns
    -------
    Function that accepts coordinates (x, y) of any point in the plane as a
    2-Iterable, and returns the interpolated value at that point.

    Notes
    -----
    Instead of single floats, the ``anchorvalues`` may also be iterables of uniform
    length, e.g. 3- or 4-tuples specifying colors. The interpolation function's
    return value has the same length as the anchorvalues.

    Source: https://stackoverflow.com/questions/57863618/how-to-vectorize-calculation-of-barycentric-coordinates-in-python
    """
    anchorpoints = np.array(anchorpoints)
    anchorvalues = np.array(anchorvalues)
    if len(anchorpoints) != len(anchorvalues):
        raise ValueError(
            "Parameters 'anchorpoints' and 'anchorvalues' must be of equal length."
        )
    if len(anchorpoints) < 3:
        raise ValueError("At least 3 anchorpoints must be specified.")

    # Tesselate into simplexes (individual triangles).
    delaunay = Delaunay(
        anchorpoints
    )  # each row has indices of the 3 anchorpoints that are the simplex corners.

    def interp(point):
        # Find simplex point is in.
        s = delaunay.find_simplex(
            point
        )  # simplex-index that contains point. (-1 means point is in none)
        if s > -1:  # normal point, inside the hull
            # Get barycentric coordinates of the triangle.
            b0 = delaunay.transform[s, :2].dot((point - delaunay.transform[s, 2]))
            weights = np.array([*b0, 1 - b0.sum()])  # add final coordinate / weight.
            indices = delaunay.simplices[s]
        else:  # point outside the hull
            if outsidevalue:
                return outsidevalue
            # Find the 2 anchorpoints on the hull line nearest to the point
            hull = ConvexHull(
                [*anchorpoints, point], qhull_options="QG" + str(len(anchorpoints))
            )
            visible = hull.simplices[
                hull.good
            ]  # lines (anchorpoints) visible from the point
            for indices in visible:  # anchor-indices of visible line
                p01 = point - anchorpoints[indices]  # from line anchors to point
                lin = anchorpoints[indices[0]] - anchorpoints[indices[1]]
                dot12 = p01.dot(lin)
                if (
                    np.sign(dot12).sum() == 0
                ):  # inside line 'shadow' if one dot product <0, >0
                    lens = np.linalg.norm(p01, axis=1)
                    lens = np.abs(dot12)
                    weights = np.flip(lens) / lens.sum()
                    break
            else:  # not in shadow of line - use value of nearest anchor.
                # Find nearest anchor (="anchor 0"). Must be included in 2 lines.
                indices = list(set(visible.flatten()))
                sd = ((anchorpoints[indices] - point) ** 2).sum(
                    axis=1
                )  # squared distance to each anchorpoint
                indices = [indices[np.argmin(sd)]]  # keep only nearest one
                weights = [1]

        # Get interpolated value.
        value = np.dot(anchorvalues[indices].T, weights)
        return value

    interp.delaunay = delaunay  # attach to function for debugging/visualisation
    return interp


def polygons(anchorpoints: Iterable, anchorvalues: Iterable) -> Callable:
    """
    Generalised barycentric interpolation (using polygons) near set of points.

    Interpolate the value at any point in the plane, if the coordinates and
    values of a set of anchors is specified. The convex hull around them is
    tessellated with polygons.

    Parameters
    ----------
    anchorpoints
        Iterable of anchor point coordinates (x, y).
    anchorvalues
        Iterable of floats specifying the known value at each anchor point.

    Returns
    -------
    Function that accepts coordinates (x, y) of any point in the plane as a
    2-Iterable, and returns the interpolated value at that point.

    Notes
    -----
    Instead of single floats, the ``anchorvalues`` may also be iterables of uniform
    length, e.g. 3- or 4-tuples specifying colors. The interpolation function's
    return value has the same length as the anchorvalues.

    For points that lie outside the hull, the interpolated value may be smaller
    (larger) than the minimum (maximum) of the specified ``anchorvalues``. This
    means that, when using colors for the ``anchorvalues``, an invalid color may
    be returned.
    """
    anchorpoints = np.array(anchorpoints)
    anchorvalues = np.array(anchorvalues)
    if len(anchorpoints) != len(anchorvalues):
        raise ValueError(
            "Parameters 'anchorpoints' and 'anchorvalues' must be of equal length."
        )
    if len(anchorpoints) < 3:
        raise ValueError("At least 3 anchorpoints must be specified.")

    # Tesselate into polygons.
    pg = Polygonate(anchorpoints, convex=False)
    # Interpolation function for each polygon...
    interpf = [polygon(anchorpoints[shape], anchorvalues[shape]) for shape in pg.shapes]
    # ...and inter(extra)polation function for the hull.
    hull = ConvexHull(anchorpoints).vertices
    interpf_hull = polygon(anchorpoints[hull], anchorvalues[hull])

    def interp(point):
        # Find simplex point is in.
        s = pg.find_shape(
            point
        )  # simplex that contains point. (-1 means point is in none)
        if s > -1:  # normal point, inside the hull
            return interpf[s](point)
        else:  # point outside the hull
            return interpf_hull(point)

    interp.polygonate = pg  # attach to function for debugging/visualitation
    return interp
