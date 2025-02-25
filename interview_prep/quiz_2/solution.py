import numpy as np

import numpy as np


def process_numpy_data(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Processes a NumPy dataset by normalizing, handling missing values, and computing statistics.

    :param data: NumPy 2D array representing the dataset.
    :return: (processed_data, column_means, column_stddevs)
    """

    col_means = np.nanmean(data, axis=0)
    col_means = np.where(np.isnan(col_means), 0, col_means)  # Handle all-NaN columns
    col_stdevs = np.nanstd(data, axis=0)
    nan_indices = np.where(np.isnan(data))
    data[nan_indices] = np.take(col_means, nan_indices[1])

    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)

    data_range = data_max - data_min
    data_range[data_range == 1] = 1

    norm_data = (data - data_min) / (data_max - data_min)

    return norm_data, col_means, col_stdevs


if __name__ == "__main__":
    data = np.array([
        [10, 200, np.nan],
        [15, np.nan, 50],
        [20, 180, 60],
        [25, 220, 70]
    ])

    output = process_numpy_data(data)


