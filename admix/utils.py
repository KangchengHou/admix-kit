"""Common utility functions for `admix`. Used by more than 2 modules"""
import admix
import os
from contextlib import contextmanager
import os
from typing import List
import glob
import hashlib


def str2int(s: str) -> int:
    """Convert string to int

    Parameters
    ----------
    s : str
        string to convert

    Returns
    -------
    int
        string and hashed
    """
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16) % (2 ** 32)


@contextmanager
def cd(newdir):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)


def get_cache_dir() -> str:
    """Get the cache directory for admix-kit

    Returns
    -------
    [type]
        [description]
    """
    cache_dir = os.path.join(os.path.dirname(admix.__file__), "../.admix_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir
