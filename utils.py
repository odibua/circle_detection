import ipdb
import numpy as np
import os
from skimage.draw import circle_perimeter_aa
from shapely.geometry import Point, GeometryCollection
from typing import Tuple


def create_log_files(log_file: str, input: str):
    """
    Create log file with input as comma separated first row

    :param log_file: Path to log file
    :param input: Comma separated first row
    :return: None
    """
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


def iou(params0, params1):
    row0, col0, rad0 = params0
    row1, col1, rad1 = params1

    shape0 = Point(row0, col0).buffer(rad0)
    shape1 = Point(row1, col1).buffer(rad1)

    return (
            shape0.intersection(shape1).area /
            shape0.union(shape1).area
    )


def normalize_gaussian(x: np.ndarray, mn: np.ndarray, std: np.ndarray):
    """
    Normalize input based on mean and standard deviation

    :param x: Input parameter
    :param mn: Mean used for normalization
    :param std: Standard deviation used for normalization
    :return: Normalized input
    """
    return (x-mn)/std


def normalize_min_max(x: np.ndarray, max: np.ndarray, min: np.ndarray):
    """
    Normalize input based on minimum and maximum value

    :param x: Input to be normalized
    :param max: Maximum value used for normalization
    :param min: Minimum value used for normalizaton
    :return: Normalized input
    """
    return 2*x/(max-min)-1


def generate_training_data(n: int, train_perc: float=0.9, mn: np.ndarray = None, std: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, float, float]:
    np.random.seed(0)
    def _list_of_tuples(list1, list2):
        return list(map(lambda x, y: (x, y), list1, list2))
    params_list = []
    image_list = []
    for i in range(n):
        params, img = noisy_circle(200, 50, 2)
        params_list.append(params)
        image_list.append(np.expand_dims(img, axis=0))

    # Split data to train and val
    train_params_list, val_params_list = np.array(params_list[0:int(n * train_perc)]), np.array(params_list[int(n * train_perc):])
    train_image_list, val_image_list = np.array(image_list[0:int(n * train_perc)]), np.array(image_list[int(n * train_perc):])

    # Normalize output to make optimization simpler
    if not isinstance(mn, np.ndarray):
        mn, std = np.mean(train_params_list, axis=0), np.std(train_params_list, axis=0)
    train_params_list = normalize_gaussian(train_params_list, mn, std)
    val_params_list = normalize_gaussian(val_params_list, mn, std)

    # Output train and test data
    train_params_list = list(map(tuple, train_params_list))
    val_params_list  = list(map(tuple, val_params_list))
    train_data = _list_of_tuples(train_params_list, train_image_list)
    val_data = _list_of_tuples(val_params_list, val_image_list)

    return train_data, val_data, mn, std


