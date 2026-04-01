from __future__ import annotations
import os
import pickle
import numpy as np


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_npy(path: str, arr: np.ndarray) -> None:
    ensure_dir(os.path.dirname(path))
    np.save(path, arr)


def load_npy(path: str) -> np.ndarray:
    return np.load(path)


def save_pickle(path: str, obj) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)