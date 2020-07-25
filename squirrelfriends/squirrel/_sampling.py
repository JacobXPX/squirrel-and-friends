import pandas as pd
import numpy as np
from sklearn import neighbors
import random
from math import ceil


def smote_sampling(data, features, label, k=5,
                   classes={"maj": 0, "min": 1},
                   percentages={"over": 9, "under": 0.5}):
    """Implement smote sampling.

    Args:
      data (DataFrame): training data.
      features (list of str): column names of feature to be used in knn.
      label (str): column name of.
      k (int): `k`-nn.
      classes (dict): major and minor classes.
      percentages (dict): the fraction of up and under sampling.

    Returns:
      result (DataFrame): result of smote sampling.
    """

    random.seed(100)

    if percentages["under"] > 1 or percentages["under"] < 0.1:
        raise ValueError("Percentage Under must be in range 0.1 - 1")
    if percentages["over"] < 1:
        raise ValueError("Percentage Over must be in at least 1")

    data = data[[label, *features]].copy()

    data_min = data.loc[data[label] == classes["min"]].reset_index()
    data_maj = data.loc[data[label] == classes["maj"]].reset_index()

    # Train knn
    data_min_features = data_min[features]
    nbrs = (neighbors.NearestNeighbors(n_neighbors=k, algorithm="auto")
                     .fit(data_min_features))
    neighbours = nbrs.kneighbors(data_min_features)[1]
    # Upsampling by knn
    new_rows = []
    for i in range(len(data_min_features)):
        for _ in range(ceil(percentages["over"] - 1)):
            chance = 1 - percentages["over"] / ceil(percentages["over"])
            if random.uniform(0, 1) >= chance:
                # Randomly pick a neighbour.
                neigh = neighbours[i][random.randint(0, k - 1)]
                diff = data_min_features.loc[neigh] - data_min_features.loc[i]
                new_rec = data_min_features.loc[i] + random.random() * diff
                new_rows.append(new_rec.to_dict())
    new_data_min = pd.DataFrame(new_rows)
    new_data_min[label] = classes["min"]

    # Downsampling
    new_data_maj = data_maj.sample(frac=float(percentages["under"]),
                                   replace=False, random_state=1024)

    frames = [data_min, new_data_min, new_data_maj]

    result = pd.concat(frames).reset_index()[[label, *features]]

    return result[[*features, label]]
