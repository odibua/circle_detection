import ipdb
import numpy as np
import os
from skimage.draw import circle_perimeter_aa
from shapely.geometry import Point, GeometryCollection
import matplotlib.pyplot as plt


def create_log_files(log_file, input):
    if os.path.exists(log_file):
        os.remove(log_file)
        f = open(log_file, "w+")
        f.write(input)
        f.close()


def draw_circle(img, row, col, rad):
    rr, cc, val = circle_perimeter_aa(row, col, rad)
    valid = (
            (rr >= 0) &
            (rr < img.shape[0]) &
            (cc >= 0) &
            (cc < img.shape[1])
    )
    img[rr[valid], cc[valid]] = val[valid]


def noisy_circle(size, radius, noise):
    img = np.zeros((size, size), dtype=np.float)

    # Circle
    row = np.random.randint(size)
    col = np.random.randint(size)
    rad = np.random.randint(10, max(10, radius))
    draw_circle(img, row, col, rad)

    # Noise
    img += noise * np.random.rand(*img.shape)
    return (row, col, rad), img


def find_circle(img):
    # Fill in this function
    return 100, 100, 30


def iou(params0, params1):
    row0, col0, rad0 = params0
    row1, col1, rad1 = params1

    shape0 = Point(row0, col0).buffer(rad0)
    shape1 = Point(row1, col1).buffer(rad1)

    return (
            shape0.intersection(shape1).area /
            shape0.union(shape1).area
    )


def giou(params0: np.ndarray, params1: np.ndarray):
    row0, col0, rad0 = params0
    row1, col1, rad1 = params1

    shape0 = Point(row0, col0).buffer(rad0)
    shape1 = Point(row1, col1).buffer(rad1)
    collection = GeometryCollection([shape0, shape1]).convex_hull

    return (
            1.0 - (iou(params0, params1) - collection.convex_hull.difference(shape0.union(shape1)).area /
            collection.convex_hull.area)
    )


def normalize(x: np.ndarray, mn: float, std: float):
    return (x-mn)/std
