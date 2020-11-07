"""
Created on Sat Nov  7 02:52:48 2020

@author: ruud
"""

from .._core import polygon, polygons, triangles
import numpy as np
import pytest


def set_singlepolygon_float():
    anchorpoints = [[0, 0], [0, 1], [1, 1], [1, 0]]
    anchorvalues = [0, 1, 2, 1]
    testpoints = [
        ([0.5, 0.5], 1),  # inside
        ([0, 0.5], 0.5),  # on edge
        ([0, 1], 1),  # on node
        ([-0.5, 0.5], 0),  # outside
    ]
    return anchorpoints, anchorvalues, testpoints


def set_singlepolygon_iterable():
    anchorpoints = [[0, 0], [0, 1], [1, 1], [1, 0]]
    anchorvalues = [[0, 10, 5], [10, 20, -2], [-20, 40, 1.2], [10, -10, 15]]
    testpoints = [
        ([0.5, 0.5], [0, 15, 4.8]),  # inside
        ([0, 0.5], [5, 15, 1.5]),  # on edge
        ([0, 1], [10, 20, -2]),  # on node
        ([-0.5, 0.5], [10, 15, -1.8]),  # outside
    ]
    return anchorpoints, anchorvalues, testpoints


def set_singletriangle_float():
    anchorpoints = [[0, 0], [0, 1], [1, 0]]
    anchorvalues = [0, 1, 2]
    testpoints = [
        ([0.5, 0.25], 1.25),  # inside
        ([0.5, 0.5], 1.5),  # on edge
        ([0, 0.5], 0.5),  # on edge
        ([0, 1], 1),  # on node
    ]
    return anchorpoints, anchorvalues, testpoints


def set_singletriangle_iterable():
    anchorpoints = [[0, 0], [0, 1], [1, 0]]
    anchorvalues = [[0, 10, 5], [10, 20, -2], [-20, 40, 1.2]]
    testpoints = [
        ([0.5, 0.25], [-7.5, 27.5, 1.35]),  # inside
        ([0.5, 0.5], [-5, 30, -0.4]),  # on edge
        ([0, 0.5], [5, 15, 1.5]),  # on edge
        ([0, 1], [10, 20, -2]),  # on node
    ]
    return anchorpoints, anchorvalues, testpoints


def set_manytriangles_float():
    anchorpoints, anchorvalues, testpoints = set_singletriangle_float()
    # add values in periphery that shouldn't influence the value
    for _ in range(10):
        x, y = np.random.rand(2)
        x += -0.5 if x < 0.5 else 0.55
        y += -0.5 if y < 0.5 else 0.55
        anchorpoints.append([x, y])
        anchorvalues.append(np.random.rand())
    return anchorpoints, anchorvalues, testpoints


def set_manytriangles_iterable():
    anchorpoints, anchorvalues, testpoints = set_singletriangle_iterable()
    # add values in periphery that shouldn't influence the value
    for _ in range(10):
        x, y = np.random.rand(2)
        x += -0.5 if x < 0.5 else 0.55
        y += -0.5 if y < 0.5 else 0.55
        anchorpoints.append([x, y])
        anchorvalues.append(np.random.rand(3))
    return anchorpoints, anchorvalues, testpoints


@pytest.mark.parametrize("f", [polygon, polygons])
@pytest.mark.parametrize(
    "testset", [set_singlepolygon_float, set_singlepolygon_iterable]
)
def test_polygoninterpolation(f, testset):
    bench(f, *testset())


@pytest.mark.parametrize("f", [polygon, polygons, triangles])
@pytest.mark.parametrize(
    "testset", [set_singletriangle_float, set_singletriangle_iterable]
)
def test_allinterpolation(f, testset):
    bench(f, *testset())


@pytest.mark.parametrize(
    "testset", [set_manytriangles_float, set_manytriangles_iterable]
)
def test_triangleinterpolation(testset):
    bench(triangles, *testset())


def bench(f, anchorpoints, anchorvalues, testpoints):
    # testpoints: Iterable of (testcoordinates, expectedvalue)-tuples
    interp = f(anchorpoints, anchorvalues)
    for point, expected_value in testpoints:
        assert np.allclose(interp(point), expected_value)


# TODO: interpolation with many points for polygons
