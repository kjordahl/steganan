import numpy as np
from imageio.v3 import imwrite
from numpy.testing import assert_array_equal

from steganan import io, steganan


def test_file_roundtrip(tmp_path):
    outfile = tmp_path / "test.tiff"
    test_arr = np.random.randint(2**51, size=(128, 128), dtype=np.uint64)
    io.save_as_nan(test_arr, outfile)
    loaded = io.load(outfile)
    assert_array_equal(test_arr, loaded)


def test_file_stacked(tmp_path):
    outfile = tmp_path / "test.tiff"
    depth = 3
    test_arr = np.random.randint(255, size=(4, 4, depth), dtype=np.uint8)
    secret = steganan.encode_array(test_arr, stack=True)
    assert np.all(np.isnan(secret))
    imwrite(outfile, secret, compression="deflate")
    loaded = io.load(outfile, stack=True, depth=depth)
    assert_array_equal(test_arr, loaded)
