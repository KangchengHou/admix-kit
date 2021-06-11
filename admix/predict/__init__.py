import matplotlib.pyplot as plt
import numpy as np
from matplotlib import collections as mc
import pandas as pd
import xarray as xr
from sklearn.linear_model import RidgeCV

__all__ = ["train", "predict"]


def train(train_dset: xr.Dataset, method: str = "ridge"):
    """Train a phenotype prediction model

    Parameters
    ----------
    train_dset : xr.Dataset
        [description]
    method : str, optional
        [description], by default "ridge"
    """
    model = RidgeCV(cv=5)
    pass
    return model


def predict(sample_dset: xr.Dataset, model):
    """Predict the phenotype with a trained model

    Parameters
    ----------
    sample_dset : xr.Dataset
        [description]
    clf : [type]
        [description]
    """
    pass
