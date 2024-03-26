import numpy as np

from src.finml_utils.vectors import normalize_vector


def test_vector_normalization():
    v = np.array([1, 2, 3])
    norm_v = normalize_vector(v)
    assert np.all([e <= 1 and e >= 0 for e in norm_v])

    v = np.array([0, 0, 0])
    norm_v = normalize_vector(v)
    assert np.all([e <= 1 and e >= 0 for e in norm_v])

    v = np.array([1, 1, 1])
    norm_v = normalize_vector(v)
    assert np.all([e <= 1 and e >= 0 for e in norm_v])
