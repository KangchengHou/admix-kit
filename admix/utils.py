"""Common utility functions for `admix`. Used by more than 2 modules"""
import admix
import os
from contextlib import contextmanager
import os
from typing import List
import glob


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
