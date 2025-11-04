import pytest
import numpy as np

from gigablochs import bloch, utils

@pytest.mark.parametrize(
    "magnetization, T1, T2, expected_shape",
    [
        # Test case 1: 1D magnetization array
        (
            np.array([1, 2, 3]),
            1.0,
            0.5,
            (3,),
        ),
        # Test case 2: 2D magnetization array
        (
            np.array([[1, 2, 3], [4, 5, 6]]),
            np.array([1.0, 2.0]),
            np.array([0.5, 1.0]),
            (2, 3),
        ),
        # Test case 3: 3D magnetization array
        (
            np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]),
            np.array([1.0, 2.0]),
            np.array([0.5, 1.0, 1.5]),
            (2, 3, 2, 2, 3),
        ),
    ],
)
def test_relax(magnetization, T1, T2, expected_shape):
    result = bloch.relax(magnetization, T1, T2, 0.001)
    assert result.shape == expected_shape

def test_relax_unextended_shapes():
    dt = 0.001
    mag = np.random.random((4, 50, 3))
    T1 = np.linspace(0.3, 2.3, 200).reshape(4, -1)[..., np.newaxis]
    T2 = np.linspace(0.05, 1.1, 200).reshape(4, -1)[..., np.newaxis]

    extented = bloch.relax(mag, T1[..., 0], T2[..., 0], dt)
    unextended = bloch.relax(mag, T1, T2, dt, extend_shapes=False)
    np.testing.assert_array_equal(extented, unextended)

def test_relax_match_shapes():
    dt = 0.001
    mag = np.random.random((4, 50, 3))
    T1 = np.linspace(0.3, 2.3, 200).reshape(4, -1)
    T2 = np.linspace(0.05, 1.1, 200).reshape(4, -1)

    match = bloch.relax(mag, T1, T2, dt, match_shapes=True)
    unmatch = bloch.relax(mag, T1, T2, dt, match_shapes=False)
    assert match.shape == (4, 50, 3)
    assert unmatch.shape == (4, 50, 4, 50, 3)

def test_relax_axis():
    dt = 0.001
    mag = np.random.random((4, 3, 50))
    T1 = np.linspace(0.3, 2.3, 4)
    T2 = np.linspace(0.05, 1.1, 4)

    result = bloch.relax(mag, T1, T2, dt, axis=1)
    assert result.shape == (4, 3, 50)

def test_relax_integration():
    dt = 0.001
    duration = 5
    T1 = np.linspace(0.3, 2.3, 20)
    T2 = np.linspace(0.05, 1.1, 30)
    mag = np.random.random((5, 3))
    mags = np.array([mag := bloch.relax(mag, T1, T2, dt) for _ in range(round(duration / dt))])
    assert mags.shape == (5000, 20, 30, 5, 3)

@pytest.mark.parametrize(
    "a, b, keepdims, axis, expected",
    [
        # Test case 1: 1D arrays
        (
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
            False,
            -1,
            32,
        ),
        # Test case 2: 2D arrays
        (
            np.array([[1, 2, 3], [4, 5, 6]]),
            np.array([[7, 8, 9], [10, 11, 12]]),
            True,
            0,
            np.array([[47, 71, 99]]),
        ),
        # Test case 3: 3D arrays
        (
            np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]),
            np.array([[[13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24]]]),
            False,
            2,
            np.array([[86, 257], [482, 761]]),
        ),
    ],
)
def test_dot(a, b, keepdims, axis, expected):
    result = utils.dot(a, b, keepdims=keepdims, axis=axis)
    np.testing.assert_array_equal(result, expected)
