"""
Encode integer array values in unused bits of floating point NaN values

"""

import numpy as np
from numpy.typing import DTypeLike, NDArray

MAX_VAL_FLOAT32 = 2**22 - 1
MAX_VAL_FLOAT64 = 2**51 - 1
MIN_VAL = 0
NAN_MASK_32 = np.float32(np.nan).view(np.uint32)
NAN_MASK_64 = np.float64(np.nan).view(np.uint64)


def encode_array(a: NDArray, stack: bool = False, dtype: DTypeLike = np.float64) -> NDArray:
    """Encode an numpy array into the payload of NaN values"""
    dtype = np.dtype(dtype)
    if dtype == np.float32:
        max_val = MAX_VAL_FLOAT32
        nan_mask: DTypeLike = NAN_MASK_32
    elif dtype == np.float64:
        max_val = MAX_VAL_FLOAT64
        nan_mask = NAN_MASK_64
    else:
        raise TypeError(f"Unsupported output data type {dtype}")
    if (a.min() < MIN_VAL) or (a.max() > max_val):
        raise ValueError(f"Input values outside the range [{MIN_VAL}, {max_val}] are not supported")
    if stack:
        if a.dtype != np.uint8:
            raise ValueError("stack=True is only supported for input dtype=np.uint8")
        if dtype != np.float64:
            raise ValueError("stack=True is only supported for output dtype=np.float64")
        if a.shape[-1] != 3:
            raise NotImplementedError("Only arrays with depth 3 are currently supported")
        a = a[..., 0].astype(np.uint32) * 256**2 + a[..., 1].astype(np.uint32) * 256 + a[..., 2]
    else:
        if dtype == np.float32 and np.iinfo(a.dtype).bits > 32:
            a = a.astype(np.uint32)
    return (a | nan_mask).view(dtype)


def decode_array(a: NDArray, stack: bool = False, depth: int = 3) -> NDArray:
    """Decode payload of NaN values into an output array"""
    if a.dtype == np.float32:
        i_dtype: DTypeLike = np.uint32
        nan_mask: DTypeLike = NAN_MASK_32
    else:
        i_dtype = np.uint64
        nan_mask = NAN_MASK_64
    decoded = a.view(i_dtype) ^ nan_mask
    if stack:
        if depth != 3:
            raise NotImplementedError("Only arrays with depth 3 are currently supported")
        shape = [*list(decoded.shape), depth]
        out_arr = np.empty(shape, dtype=np.uint8)
        out_arr[..., 0] = decoded // 256**2
        out_arr[..., 1] = decoded // 256
        out_arr[..., 2] = decoded % 256
        return out_arr
    else:
        return decoded


def str_to_bytes_array(s: str) -> NDArray[np.bytes_]:
    """Convert a string to a numpy bytes_ array, one character per element"""
    return np.array([_.encode() for _ in list(s)]).astype(np.bytes_)


def str_to_uint32_array(s: str) -> NDArray[np.uint32]:
    """Convert a string to a numpy uint32 array, one character per element"""
    b = str_to_bytes_array(s)
    return np.array([int.from_bytes(_, byteorder="little") for _ in b]).astype(np.uint32)


def uint32_array_to_str(a: NDArray[np.uint32]) -> str:
    """Convert a numpy uint32 array representing characters to a Python string"""
    if a.dtype != np.uint32:
        raise TypeError("Expected a uint32 array")
    c = [_.view("|S4").decode() for _ in a]
    return "".join(c)


def write_str_to_nans(a: NDArray, s: str) -> None:
    """Write characters from a string into NaN payloads in the input array.
    There must be sufficient NaN values already existing in a.
    Locations will be chosen at random.

    array a: Input floating point array
    str s: String to encode into NaN payloads
    """
    i, j = np.nonzero(np.isnan(a))
    if len(i) < len(s):
        raise ValueError(f"Not enough NaN values in input array ({len(i)}) to store input string of len {len(s)}")
    idx = sorted(np.random.choice(len(i), len(s), replace=False))
    i_vals = i[idx]
    j_vals = j[idx]
    b = str_to_uint32_array(s)
    a[i_vals, j_vals] = encode_array(b, dtype=a.dtype)


def retrieve_string_from_payloads(a: NDArray) -> str:
    """Extract a string from encoded NaN payload values in input array"""
    p = a[is_payload_nan(a)]
    b = decode_array(p).astype(np.uint32)
    return uint32_array_to_str(b)


def is_payload_nan(a: NDArray) -> NDArray:
    """Find NaN values that contain paylads in an array"""
    if a.dtype == np.float32:
        dtype: DTypeLike = np.uint32
        nan_mask: DTypeLike = NAN_MASK_32
    elif a.dtype == np.float64:
        dtype = np.uint64
        nan_mask = NAN_MASK_64
    else:
        raise TypeError("Only float32 and float64 data types are currently supported")
    nan_idx = np.isnan(a)
    nan_mask_idx = (a.view(dtype) ^ nan_mask) == 0
    idx: NDArray[np.bool] = nan_idx & ~nan_mask_idx
    return idx


if __name__ == "__main__":
    test_arr = np.random.randint(1000, size=(4, 4), dtype=np.uint64)
    print("input:\n", test_arr)
    secret = encode_array(test_arr)
    print("encoded:\n", secret)
    print("decoded:\n", decode_array(secret))
