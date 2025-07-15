# steganan

[![PyPI - Version](https://img.shields.io/pypi/v/steganan)](https://pypi.org/project/steganan/)
[![Release](https://img.shields.io/github/v/release/kjordahl/steganan)](https://github.com/kjordahl/steganan/releases)
[![Build status](https://img.shields.io/github/actions/workflow/status/kjordahl/steganan/main.yml?branch=main)](https://github.com/kjordahl/steganan/actions/workflows/main.yml?query=branch%3Amain)
[![License](https://img.shields.io/github/license/kjordahl/steganan)](https://github.com/kjordahl/steganan/blob/main/LICENSE)

Steganography in floating point data with NaN payloads

- **Github repository**: <https://github.com/kjordahl/steganan/>

Do you have not-a-number values in your floating point data that
you're not making full use of?  Have you ever pondered what to do with
all of those wasted bits? Steganan may be for you!  Encode whatever
data you like into those spare bits! Most programs won't even notice
the difference. Save them out to a file if you like. As long as your
file format preserves the full floating point values, you can get your
hidden data back!

Want to add compression and/or encryption of your data? That is
certainly possible, but is left as an exercise for the user.

## Examples

### Storing image data in an array of NaN values

```
>>> import matplotlib.pyplot as plt
>>> import numpy as np
>>> from skimage.data import astronaut
>>> import steganan
>>> img = astronaut()
>>> print(img.shape)
(512, 512, 3)
>>> a = steganan.encode_array(img, stack=True, dtype=np.float64)
>>> print(a)
[[nan nan nan ... nan nan nan]
 [nan nan nan ... nan nan nan]
 [nan nan nan ... nan nan nan]
 ...
 [nan nan nan ... nan nan nan]
 [nan nan nan ... nan nan nan]
 [nan nan nan ... nan nan nan]]
>>> decoded = steganan.decode_array(a, stack=True, depth=3)
>>> plt.figure(figsize=(4, 4))
>>> plt.imshow(decoded)
```
![decoded.png](https://github.com/kjordahl/steganan/raw/refs/heads/main/data/decoded.png)

### Hiding data in an existing floating point array

```
>>> import rasterio as rio
>>> src = rio.open("data/modis_aod_06_2025.tif")
>>> a = src.read(1)
>>> steganan.write_str_to_nans(a, "I have a secret!")
>>> plt.imshow(a, cmap='inferno_r', vmin=0, vmax=1.0)
```
![encoded.png](https://github.com/kjordahl/steganan/raw/refs/heads/main/data/encoded.png)
```
>>> message = steganan.retrieve_string_from_payloads(a)
>>> print(message)
I have a secret!
```

### Notebook example

See [lightning talk slides](https://github.com/kjordahl/steganan/blob/main/slides/talk.ipynb).

---

Repository initiated with [fpgmaas/cookiecutter-uv](https://github.com/fpgmaas/cookiecutter-uv).
