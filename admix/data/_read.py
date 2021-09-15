import pandas as pd
import numpy as np
import re
from smart_open import open


def read_vcf(path: str):
    pass


def read_digit_mat(path, filter_non_numeric=False):
    """
    Read a matrix of integer with [0-9], and with no delimiter.

    Args
    ----

    """
    if filter_non_numeric:
        with open(path) as f:
            mat = np.array(
                [
                    np.array([int(c) for c in re.sub("[^0-9]", "", line.strip())])
                    for line in f.readlines()
                ],
                dtype=np.int8,
            )
    else:
        with open(path) as f:
            mat = np.array(
                [np.array([int(c) for c in line.strip()]) for line in f.readlines()],
                dtype=np.int8,
            )
    return mat


def write_digit_mat(path, mat):
    """
    Read a matrix of integer with [0-9], and with no delimiter.

    Args
    ----

    """
    np.savetxt(path, mat, fmt="%d", delimiter="")
