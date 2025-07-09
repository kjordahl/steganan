import numpy as np
from numpy.testing import assert_array_equal

from steganan import steganan


def test_file_roundtrip(tmp_path):
    outfile = tmp_path / "test.tiff"
    test_arr = np.random.randint(2**51, size=(128, 128), dtype=np.uint64)
    steganan.save(test_arr, outfile)
    loaded = steganan.load(outfile)
    assert_array_equal(test_arr, loaded)
