"""
smFISH image preprocessing before dot detection.

Vectorized port of the pyfish_tools pre_processing.py workflow (Katsuya Lex Colon) as used in
JD_Preprocessing.ipynb:
    background_correct_image(correction_algo=Gaussian_and_Gamma_Correction, kern=9, sigma=6, gamma=1.6,
                             match_hist=False, subtract=True, divide=False, tophat_raw=False)
    low_pass_gaussian(kern=9)

Per z-slice and per channel:
    - FISH channels: subtract a Gaussian-blurred background (kern x kern, sigma), clip negatives to 0,
      gamma-correct (x ** gamma), cast to uint16.
    - Last channel (DAPI): radial edge-dimming correction via least-squares fit of intensity vs distance
      from the image center.
    - All channels: low-pass Gaussian blur (kern x kern, sigma=1).

Differences from the original:
    - Uses numpy/scipy only (no cv2/sklearn/photutils), so it runs in the wormfish_analysis env.
      Gaussian kernels and border handling (reflect-101) match cv2.GaussianBlur.
    - Values above 65535 after gamma correction are clipped instead of wrapping around in the uint16 cast.
    - Streams one z-slice at a time, so memory stays low for large stacks.
"""
import os
import glob
import numpy as np
import tifffile
from scipy.ndimage import correlate1d

from HotPixel_functions import find_hot_pixels, correct_hot_pixels, hot_pixel_file, save_hot_pixels


def gaussian_kernel_1d(kern, sigma):
    """1D Gaussian kernel equivalent to cv2.getGaussianKernel(kern, sigma)."""
    x = np.arange(kern) - (kern - 1) / 2
    k = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def gaussian_blur(image, kern, sigma):
    """Separable Gaussian blur matching cv2.GaussianBlur(image, (kern, kern), sigma) on float input."""
    k = gaussian_kernel_1d(kern, sigma)
    out = correlate1d(image, k, axis=-1, mode="mirror")
    return correlate1d(out, k, axis=-2, mode="mirror")


def gaussian_and_gamma_correction(image, gamma=1.6, sigma=6, kern=9):
    """Subtract Gaussian-blurred background, clip negatives, gamma-correct. Returns uint16."""
    image = image.astype(np.float64)
    corrected = image - gaussian_blur(image, kern, sigma)
    corrected[corrected < 0] = 0
    corrected = corrected ** gamma
    return np.clip(corrected, 0, 65535).astype(np.uint16)


def lsr_background_correction(image):
    """Radial edge-dimming correction via least-squares fit of intensity vs distance from center. Returns uint16."""
    image = image.astype(np.float64)
    if not image.any():  # blank slice (e.g. padding at the end of a converted stack)
        return np.zeros(image.shape, dtype=np.uint16)
    xx, yy = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]))
    center_x, center_y = int(image.shape[0] / 2), int(image.shape[1] / 2)
    dist = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2).ravel()
    slope, intercept = np.polyfit(dist, image.ravel(), 1)
    pred = dist * slope + intercept
    corrected = image.ravel() / (pred / pred.mean())
    return np.clip(corrected, 0, 65535).astype(np.uint16).reshape(image.shape)


def low_pass_gaussian(image, kern=9, sigma=1):
    """Low-pass Gaussian blur of a uint16 2D image, rounded back to uint16 (as cv2 does)."""
    blurred = gaussian_blur(image.astype(np.float64), kern, sigma)
    return np.clip(np.rint(blurred), 0, 65535).astype(np.uint16)


def preprocess_slice(slice_cyx, gamma=1.6, sigma=6, kern=9, lowpass_kern=9, dapi_last=True):
    """
    Preprocess one z-slice of shape (C, Y, X).
    FISH channels get Gaussian background subtraction + gamma; the last channel (DAPI) gets LSR correction.
    All channels then get a low-pass Gaussian blur.
    """
    n_ch = slice_cyx.shape[0]
    out = np.empty(slice_cyx.shape, dtype=np.uint16)
    for c in range(n_ch):
        if dapi_last and c == n_ch - 1:
            corrected = lsr_background_correction(slice_cyx[c])
        else:
            corrected = gaussian_and_gamma_correction(slice_cyx[c], gamma=gamma, sigma=sigma, kern=kern)
        out[c] = low_pass_gaussian(corrected, kern=lowpass_kern)
    return out


def preprocess_tiff_file(input_path, output_path, gamma=1.6, sigma=6, kern=9, lowpass_kern=9, dapi_last=True,
                         hot_pixel_correction=False):
    """
    Preprocess a single (Z, C, Y, X) TIFF stack slice by slice and write it as an ImageJ hyperstack.
    If hot_pixel_correction is True, camera hot pixels are found (see HotPixel_functions), saved next to the
    output as <name>_hotpixels.csv, and replaced by the median of their neighbours before preprocessing.
    Returns the output path, or None if the file is not a (Z, C, Y, X) stack.
    """
    with tifffile.TiffFile(input_path) as tif:
        series = tif.series[0]
        if series.axes != "ZCYX":
            print(f"  Skipping {os.path.basename(input_path)}: axes {series.axes}, shape {series.shape} (expected ZCYX)")
            return None
        n_z, n_c = series.shape[:2]
        pages = series.pages

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        hot_pixels = np.zeros((0, 2), dtype=int)
        if hot_pixel_correction:
            hot_pixels = find_hot_pixels(input_path)
            save_hot_pixels(hot_pixel_file(output_path), hot_pixels)
            print(f"  Hot pixels corrected: {len(hot_pixels)}")

        out = tifffile.memmap(output_path, shape=series.shape, dtype=np.uint16,
                              imagej=True, metadata={"axes": "ZCYX"})
        for z in range(n_z):
            slice_cyx = np.stack([pages[z * n_c + c].asarray() for c in range(n_c)])
            if not slice_cyx.any():  # blank padding plane stays blank
                out[z] = 0
                continue
            slice_cyx = correct_hot_pixels(slice_cyx, hot_pixels)
            out[z] = preprocess_slice(slice_cyx, gamma=gamma, sigma=sigma, kern=kern,
                                      lowpass_kern=lowpass_kern, dapi_last=dapi_last)
        out.flush()
        del out

    print(f"  ✓ {os.path.basename(input_path)} -> {output_path} | shape={series.shape}")
    return output_path


def preprocess_folders(input_root, output_root=None, gamma=1.6, sigma=6, kern=9, lowpass_kern=9,
                       dapi_last=True, hot_pixel_correction=False, overwrite=False):
    """
    Preprocess every .tif in each subfolder of input_root (and in input_root itself).
    Output mirrors the folder structure under output_root (default: <input_root>_Preprocessed).

    Args:
        input_root: Root directory containing subfolders of (Z, C, Y, X) TIFF stacks
        output_root: Root directory to save preprocessed stacks
        gamma: Gamma correction exponent for FISH channels
        sigma: Sigma of the Gaussian background estimate
        kern: Kernel size of the Gaussian background estimate
        lowpass_kern: Kernel size of the final low-pass Gaussian (sigma=1)
        dapi_last: Treat the last channel as DAPI (LSR correction instead of background subtraction)
        hot_pixel_correction: Find and correct camera hot pixels before preprocessing (saved as *_hotpixels.csv)
        overwrite: Reprocess files whose output already exists
    """
    if output_root is None:
        output_root = os.path.normpath(input_root) + "_Preprocessed"

    print("=" * 60)
    print("smFISH Preprocessing")
    print("=" * 60)
    print(f"Background: kern={kern}, sigma={sigma}, gamma={gamma} | low-pass kern={lowpass_kern} | "
          f"hot pixel correction: {hot_pixel_correction}")
    print(f"Input: {input_root}")
    print(f"Output: {output_root}")

    tiff_files = sorted(glob.glob(os.path.join(input_root, "*.tif")) +
                        glob.glob(os.path.join(input_root, "*", "*.tif")))
    tiff_files = [f for f in tiff_files if not os.path.normpath(f).startswith(os.path.normpath(output_root))]
    print(f"\nFound {len(tiff_files)} .tif files")

    created = []
    for i, tiff_file in enumerate(tiff_files, 1):
        rel_path = os.path.relpath(tiff_file, input_root)
        output_path = os.path.join(output_root, rel_path)
        print(f"\n[{i}/{len(tiff_files)}] {rel_path}")
        if os.path.exists(output_path) and not overwrite:
            print("  Already exists, skipping")
            created.append(output_path)
            continue
        result = preprocess_tiff_file(tiff_file, output_path, gamma=gamma, sigma=sigma, kern=kern,
                                      lowpass_kern=lowpass_kern, dapi_last=dapi_last,
                                      hot_pixel_correction=hot_pixel_correction)
        if result:
            created.append(result)

    print("\n" + "=" * 60)
    print(f"✓ Preprocessed {len(created)} files. Results saved in: {output_root}")
    print("=" * 60)
    return created
