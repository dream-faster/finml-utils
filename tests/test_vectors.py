import numpy as np

from src.finml_utils.vectors import normalize_vector


def test_vector_normalization():
    # Test case 1
    v = np.array([1, 2, 3])
    norm_v = normalize_vector(v)
    assert np.all([e <= 1 and e >= 0 for e in norm_v])

    # Test case 2
    v = np.array([0, 0, 0])
    norm_v = normalize_vector(v)
    assert np.all([e <= 1 and e >= 0 for e in norm_v])

    # Test case 3
    v = np.array([1, 1, 1])
    norm_v = normalize_vector(v)
    assert np.all([e <= 1 and e >= 0 for e in norm_v])
