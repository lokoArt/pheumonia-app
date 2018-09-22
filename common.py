import numpy as np


def get_label_matrix_by_folder(folder):
    return np.array([0, 1]) if folder == 'NORMAL' else np.array([1, 0])


def get_label_by_matrix(matrix):
    return 'NORMAL' if np.argmax(matrix) == 1 else 'PNEUMONIA'
