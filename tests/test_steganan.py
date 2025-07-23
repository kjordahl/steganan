import numpy as np
import pytest
from numpy.testing import assert_array_equal

import steganan


def test_encode_roundtrip():
    for dtype in [np.uint8, np.uint16, np.uint32, np.uint64]:
        max_val = min(2**51, np.iinfo(dtype).max)
        test_arr = np.random.randint(max_val, size=(128, 128), dtype=np.uint64)
        secret = steganan.encode_array(test_arr)
        assert np.all(np.isnan(secret))
        assert_array_equal(test_arr, steganan.decode_array(secret))


def test_encode_toobig():
    test_arr = 2**51 * np.ones((2), dtype=np.uint64)
    with pytest.raises(ValueError):
        _ = steganan.encode_array(test_arr)
    test_arr_32 = 2**22 * np.ones((2), dtype=np.uint32)
    with pytest.raises(ValueError):
        _ = steganan.encode_array(test_arr_32, dtype=np.float32)


def test_encode_roundtrip32():
    for dtype in [np.uint8, np.uint16, np.uint32, np.uint64]:
        max_val = min(2**22, np.iinfo(dtype).max)
        test_arr = np.random.randint(max_val, size=(128, 128), dtype=dtype)
        secret = steganan.encode_array(test_arr, dtype=np.float32)
        assert secret.dtype == np.float32
        assert np.all(np.isnan(secret))
        output_arr = steganan.decode_array(secret)
        assert output_arr.dtype == np.uint32
        assert_array_equal(test_arr, output_arr)


def test_encode_stacked_roundtrip():
    depth = 3
    test_arr = np.random.randint(255, size=(4, 4, depth), dtype=np.uint8)
    secret = steganan.encode_array(test_arr, stack=True)
    assert np.all(np.isnan(secret))
    assert_array_equal(test_arr, steganan.decode_array(secret, stack=True, depth=depth))


def test_encode_stacked_32_raises():
    depth = 3
    test_arr = np.random.randint(255, size=(4, 4, depth), dtype=np.uint8)
    with pytest.raises(ValueError):
        _ = steganan.encode_array(test_arr, stack=True, dtype=np.float32)


def test_str_to_bytes_array():
    test_string = "île à côté 🙂"
    a = steganan.str_to_bytes_array(test_string)
    res_string = "".join([_.decode() for _ in a])
    assert test_string == res_string


def test_str_to_uint32_array():
    test_string = "île à côté"
    a = steganan.str_to_uint32_array(test_string)
    res_string = steganan.uint32_array_to_str(a)
    assert test_string == res_string
    secret = steganan.encode_array(a, dtype=np.float32)
    assert np.all(np.isnan(secret))
    decoded = steganan.decode_array(secret)
    res_string = steganan.uint32_array_to_str(decoded)
    assert test_string == res_string
    wide_test_string = test_string + " 🙂"
    w = steganan.str_to_uint32_array(wide_test_string)
    # float64 NaN has sufficient bits to encode all UTF-8 values
    secret = steganan.encode_array(w, dtype=np.float64)
    with pytest.raises(ValueError):
        # float32 NaN does not have enough bits for 4-byte UTF-8 code points
        secret = steganan.encode_array(w, dtype=np.float32)


def test_write_str_to_nans():
    test_string = "île à côté 🙂"
    a = np.random.random([100, 100])
    with pytest.raises(ValueError):
        steganan.write_str_to_nans(a, test_string)
    i = np.random.randint(100, size=20)
    j = np.random.randint(100, size=20)
    a[i, j] = np.nan
    n_nans = np.isnan(a).sum()
    steganan.write_str_to_nans(a, test_string)
    assert np.isnan(a).sum() == n_nans
    # retrieve the stored string
    p = a[steganan.is_payload_nan(a)]
    assert len(p) == len(test_string)
    b = steganan.decode_array(p).astype(np.uint32)
    res_string = steganan.uint32_array_to_str(b)
    assert test_string == res_string
    new_res_string = steganan.retrieve_string_from_payloads(a)
    assert test_string == new_res_string


def test_write_str_to_nans_32():
    test_string = "île à côté"
    wide_test_string = "île à côté 🙂"
    a = np.random.random([100, 100]).astype(np.float32)
    with pytest.raises(ValueError):
        steganan.write_str_to_nans(a, test_string)
    i = np.random.randint(100, size=20)
    j = np.random.randint(100, size=20)
    a[i, j] = np.nan
    n_nans = np.isnan(a).sum()
    with pytest.raises(ValueError):
        # This will fail with 4-byte UTF-8 code points
        steganan.write_str_to_nans(a, wide_test_string)
    steganan.write_str_to_nans(a, test_string)
    assert np.isnan(a).sum() == n_nans
    # retrieve the stored string
    p = a[steganan.is_payload_nan(a)]
    assert len(p) == len(test_string)
    b = steganan.decode_array(p).astype(np.uint32)
    res_string = steganan.uint32_array_to_str(b)
    assert test_string == res_string
    new_res_string = steganan.retrieve_string_from_payloads(a)
    assert test_string == new_res_string


def test_is_payload_nan():
    test_array = np.random.random([10, 10])
    assert_array_equal(steganan.is_payload_nan(test_array), 0)
    test_array[:5, :5] = np.nan
    assert_array_equal(steganan.is_payload_nan(test_array), 0)
    val_arr = np.random.randint(10000, size=(5, 5), dtype=np.uint64)
    test_array[5:, 5:] = steganan.encode_array(val_arr)
    expected = np.zeros(test_array.shape, dtype=bool)
    expected[5:, 5:] = True
    assert_array_equal(steganan.is_payload_nan(test_array), expected)


def test_is_payload_nan_32():
    test_array = np.random.random([10, 10]).astype(np.float32)
    assert_array_equal(steganan.is_payload_nan(test_array), 0)
    test_array[:5, :5] = np.nan
    assert_array_equal(steganan.is_payload_nan(test_array), 0)
    val_arr = np.random.randint(10000, size=(5, 5), dtype=np.uint32)
    test_array[5:, 5:] = steganan.encode_array(val_arr, dtype=np.float32)
    expected = np.zeros(test_array.shape, dtype=bool)
    expected[5:, 5:] = True
    assert_array_equal(steganan.is_payload_nan(test_array), expected)
