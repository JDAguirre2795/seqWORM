"""
Camera hot-pixel detection and correction.

A hot pixel is a single sensor pixel that reads high in every image regardless of the sample. In a z-stack it
shows up as a bright one-pixel column through every plane and channel, which spot detection can pick up and
dense-region decomposition can turn into hundreds of stacked spots.

Detection (per stack): take the minimum over all non-blank planes of the FISH channels. A pixel is hot if that
minimum is > `factor` x the median of the minimum image AND > `neighbour_factor` x the median of its 8 neighbours
(so it is an isolated single pixel, not sample signal).
Correction: replace each hot pixel with the median of its 8 neighbours, in every plane and channel.
"""
import os
import numpy as np
import tifffile


def find_hot_pixels(input_path, factor=3, neighbour_factor=2, channels=None):
    """
    Return an (N, 2) int array of (y, x) hot pixels in a (Z, C, Y, X) TIFF stack.
    channels: channels used for detection (default: all but the last, i.e. the FISH channels; DAPI is imaged with
        a shorter exposure, so hot pixels are much dimmer there). Correction is applied to all channels.
    """
    with tifffile.TiffFile(input_path) as tif:
        series = tif.series[0]
        if series.axes != "ZCYX":
            return np.zeros((0, 2), dtype=int)
        n_z, n_c = series.shape[:2]
        if channels is None:
            channels = range(max(n_c - 1, 1))
        min_img = None
        for z in range(n_z):
            for c in channels:
                plane = series.pages[z * n_c + c].asarray()
                if not plane.any():  # blank padding plane
                    continue
                min_img = plane.copy() if min_img is None else np.minimum(min_img, plane)
    if min_img is None:
        return np.zeros((0, 2), dtype=int)

    min_img = min_img.astype(np.float32)
    candidates = np.argwhere(min_img > factor * np.median(min_img))
    hot = []
    for y, x in candidates:
        win = min_img[max(y - 1, 0):y + 2, max(x - 1, 0):x + 2].ravel().tolist()
        win.remove(min_img[y, x])
        if min_img[y, x] > neighbour_factor * np.median(win):
            hot.append((y, x))
    return np.array(hot, dtype=int).reshape(-1, 2)


def correct_hot_pixels(image, hot_pixels):
    """Replace hot pixels with the median of their 8 neighbours. Works on (..., Y, X) arrays; returns a copy."""
    if len(hot_pixels) == 0:
        return image
    out = image.copy()
    H, W = image.shape[-2:]
    for y, x in hot_pixels:
        ys = [yy for yy in (y - 1, y, y + 1) if 0 <= yy < H]
        xs = [xx for xx in (x - 1, x, x + 1) if 0 <= xx < W]
        neighbours = [image[..., yy, xx] for yy in ys for xx in xs if (yy, xx) != (y, x)]
        out[..., y, x] = np.median(np.stack(neighbours, axis=-1), axis=-1).astype(image.dtype)
    return out


def hot_pixel_file(tiff_path):
    """Path of the CSV that stores the hot pixels found for a stack (next to the stack)."""
    return os.path.splitext(tiff_path)[0] + "_hotpixels.csv"


def save_hot_pixels(path, hot_pixels):
    np.savetxt(path, hot_pixels, fmt="%d", delimiter=",", header="y,x", comments="")


def load_hot_pixels(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rows = [line.strip() for line in f.readlines()[1:] if line.strip()]
    return np.array([[int(v) for v in r.split(",")] for r in rows], dtype=int).reshape(-1, 2)
