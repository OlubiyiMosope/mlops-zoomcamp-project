import joblib
from pathlib import Path
import numpy as np

from training import abalone_flow


test_directory = Path(__file__).parent


def test_load_data():

    path = f"{test_directory}/abalone_data.csv"
    pipeline = joblib.load(f"{test_directory}/pipeline.pkl")

    *data, pipeline = abalone_flow.load_data(path, pipeline)  #

    assert type(data[0]) == np.ndarray and len(data) == 4

