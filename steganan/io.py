from imageio.v3 import imread, imwrite

from steganan import steganan


def save_as_nan(a, filename, plugin="tifffile"):
    """Save an array as an encoded TIFF file of NaNs."""
    imwrite(filename, steganan.encode_array(a), plugin=plugin, compression="deflate")


def load(filename, **kwargs):
    """Load a floating point array from file and decode."""
    return steganan.decode_array(imread(filename), **kwargs)
