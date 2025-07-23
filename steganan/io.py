from imageio.v3 import imread, imwrite
from numpy.typing import NDArray

from steganan import steganan


def save_as_nan(a: NDArray, filename: str, plugin: str = "tifffile") -> None:
    """Save an array as an encoded TIFF file of NaNs."""
    imwrite(filename, steganan.encode_array(a), plugin=plugin, compression="deflate")


def load(filename: str, stack: bool = False, depth: int = 3) -> NDArray:
    """Load a floating point array from file and decode."""
    return steganan.decode_array(imread(filename), stack=stack, depth=depth)
