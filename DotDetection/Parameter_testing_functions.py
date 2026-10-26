"""
Parameter testing for preprocessing + dot detection.

Runs preprocessing (in memory) and big-fish spot detection on a few (Z, C, Y, X) stacks with a given parameter set,
and shows the preprocessed MIP next to the same MIP with the detected dots overlaid, for each channel.
Contrast is shared across samples per channel, so a positive sample and a negative control can be compared directly.
"""
import os
import time
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter

from Preprocessing_functions import preprocess_slice
from bigfish.detection import detect_spots, decompose_dense


def load_and_preprocess(input_path, preprocess=True, gamma=1.6, sigma=6, kern=9, lowpass_kern=9, channels=(0, 1)):
    """Load a (Z, C, Y, X) stack and return the requested channels as a (C, Z, Y, X) array, preprocessed if requested."""
    with tifffile.TiffFile(input_path) as tif:
        series = tif.series[0]
        if series.axes != "ZCYX":
            raise ValueError(f"{input_path}: axes {series.axes}, expected ZCYX")
        n_z, n_c = series.shape[:2]
        out = np.empty((len(channels), n_z) + series.shape[2:], dtype=np.uint16)
        for z in range(n_z):
            slice_cyx = np.stack([series.pages[z * n_c + c].asarray() for c in range(n_c)])
            if preprocess:
                slice_cyx = preprocess_slice(slice_cyx, gamma=gamma, sigma=sigma, kern=kern, lowpass_kern=lowpass_kern)
            out[:, z] = slice_cyx[list(channels)]
    return out


def detect_channel(image, threshold, voxel_size, spot_radius, decompose=True, alpha=0.5, beta=1.7, gamma=15):
    """Detect spots in one (Z, Y, X) channel. Returns (spots before decomposition, spots after, threshold used)."""
    spots, threshold_used = detect_spots(images=image, threshold=threshold, return_threshold=True,
                                         voxel_size=voxel_size, spot_radius=spot_radius)
    spots_post = spots
    if decompose and len(spots) > 0:
        try:
            spots_post, _, _ = decompose_dense(image=image, spots=spots, voxel_size=voxel_size,
                                               spot_radius=spot_radius, alpha=alpha, beta=beta, gamma=gamma)
        except Exception as e:
            print(f"    decompose_dense failed ({type(e).__name__}: {e}), keeping undecomposed spots")
    return spots, spots_post, threshold_used


def _densest_window(mip, size):
    """Top-left corner of the size x size window with the highest mean intensity."""
    smoothed = uniform_filter(mip.astype(np.float32), size)
    half = size // 2
    inner = smoothed[half:mip.shape[0] - half, half:mip.shape[1] - half]
    y, x = np.unravel_index(np.argmax(inner), inner.shape)
    return y, x


def run_parameter_test(files, labels=None, output_dir=None,
                       preprocess=True, pre_gamma=1.6, pre_sigma=6, pre_kern=9, lowpass_kern=9,
                       voxel_size=(1000, 100.5, 100.5), spot_radius=(1050, 300, 300), decompose=True,
                       a647_threshold=15, a647_alpha=0.5, a647_beta=1.7, a647_gamma=15,
                       cy3b_threshold=25, cy3b_alpha=0.5, cy3b_beta=1.7, cy3b_gamma=25,
                       zoom_size=256, contrast_percentiles=(1, 99.9), dot_color="#ff3b30", show=True):
    """
    Preprocess + detect spots on each file with one parameter set, then plot per channel:
    preprocessed MIP | same MIP with detected dots, full field and a zoom, for every file side by side.

    Args:
        files: List of (Z, C, Y, X) TIFF paths (e.g. one positive sample and one no-probe control)
        labels: Names shown in the figure (default: file names)
        output_dir: Folder to save the figure (None = don't save)
        preprocess: Apply preprocessing before detection (False = detect on raw)
        pre_gamma, pre_sigma, pre_kern, lowpass_kern: Preprocessing parameters (see Preprocessing_functions)
        voxel_size, spot_radius: Detection geometry in nm (z, y, x)
        decompose: Run dense-region decomposition after detection
        a647_threshold, cy3b_threshold: Detection thresholds (None = big-fish automatic threshold)
        a647_alpha/beta/gamma, cy3b_alpha/beta/gamma: Decomposition parameters
        zoom_size: Side of the zoomed window in pixels (placed on the brightest region of each sample)
        contrast_percentiles: Display range, computed over all files together per channel

    Returns:
        dict {label: {channel: {"n_detected", "n_decomposed", "threshold", "spots"}}}
    """
    labels = labels or [os.path.splitext(os.path.basename(f))[0] for f in files]
    channel_params = {
        "A647": dict(index=0, threshold=a647_threshold, alpha=a647_alpha, beta=a647_beta, gamma=a647_gamma),
        "Cy3B": dict(index=1, threshold=cy3b_threshold, alpha=cy3b_alpha, beta=cy3b_beta, gamma=cy3b_gamma),
    }

    results, mips = {}, {}
    for path, label in zip(files, labels):
        t0 = time.time()
        print(f"{label}: loading{' + preprocessing' if preprocess else ''}...")
        stack = load_and_preprocess(path, preprocess=preprocess, gamma=pre_gamma, sigma=pre_sigma,
                                    kern=pre_kern, lowpass_kern=lowpass_kern)
        results[label], mips[label] = {}, {}
        for ch, p in channel_params.items():
            image = stack[p["index"]]
            spots, spots_post, thr = detect_channel(image, p["threshold"], voxel_size, spot_radius, decompose=decompose,
                                                    alpha=p["alpha"], beta=p["beta"], gamma=p["gamma"])
            results[label][ch] = {"n_detected": len(spots), "n_decomposed": len(spots_post),
                                  "threshold": float(thr), "spots": spots_post}
            mips[label][ch] = image.max(axis=0)
            print(f"  {ch}: threshold {float(thr):.1f} | detected {len(spots)} | after decomposition {len(spots_post)}")
        print(f"  done in {time.time() - t0:.0f}s")
        del stack

    # Figure: rows = (channel full, channel zoom) per channel; columns = (MIP, MIP + dots) per file
    n_files = len(files)
    fig, axs = plt.subplots(4, 2 * n_files, figsize=(4.2 * 2 * n_files, 4.3 * 4), squeeze=False)
    for ci, ch in enumerate(channel_params):
        all_px = np.concatenate([mips[l][ch].ravel()[::7] for l in labels])
        vmin, vmax = np.percentile(all_px, contrast_percentiles)
        vmax = max(vmax, vmin + 1)
        for fi, label in enumerate(labels):
            mip = mips[label][ch]
            spots = results[label][ch]["spots"]
            y0, x0 = _densest_window(mip, zoom_size)
            in_zoom = ((spots[:, 1] >= y0) & (spots[:, 1] < y0 + zoom_size) &
                       (spots[:, 2] >= x0) & (spots[:, 2] < x0 + zoom_size)) if len(spots) else np.zeros(0, bool)
            r = results[label][ch]
            for k, with_dots in enumerate((False, True)):
                ax_full, ax_zoom = axs[2 * ci, 2 * fi + k], axs[2 * ci + 1, 2 * fi + k]
                ax_full.imshow(mip, cmap="gray", vmin=vmin, vmax=vmax)
                ax_zoom.imshow(mip[y0:y0 + zoom_size, x0:x0 + zoom_size], cmap="gray", vmin=vmin, vmax=vmax)
                ax_full.add_patch(plt.Rectangle((x0, y0), zoom_size, zoom_size, fill=False, ec="#e8a33d", lw=1))
                if with_dots and len(spots):
                    ax_full.scatter(spots[:, 2], spots[:, 1], s=0.5, c=dot_color, linewidths=0)
                    zs = spots[in_zoom]
                    ax_zoom.scatter(zs[:, 2] - x0, zs[:, 1] - y0, s=14, facecolors="none", edgecolors=dot_color, linewidths=0.6)
                title = (f"{label} | {ch} + dots\n{r['n_decomposed']} spots ({r['n_detected']} before decomp.), thr {r['threshold']:.0f}"
                         if with_dots else f"{label} | {ch} {'preprocessed' if preprocess else 'raw'} MIP")
                ax_full.set_title(title, fontsize=9)
                ax_zoom.set_title(f"zoom {zoom_size}px" + (f": {in_zoom.sum()} spots" if with_dots else ""), fontsize=8)
                for ax in (ax_full, ax_zoom):
                    ax.set_xticks([]); ax.set_yticks([])

    param_text = (f"preprocess={preprocess} (gamma={pre_gamma}, sigma={pre_sigma}, kern={pre_kern}, lowpass={lowpass_kern}) | "
                  f"voxel={voxel_size}, radius={spot_radius}, decompose={decompose} | "
                  f"A647 thr={a647_threshold} a/b/g={a647_alpha}/{a647_beta}/{a647_gamma} | "
                  f"Cy3B thr={cy3b_threshold} a/b/g={cy3b_alpha}/{cy3b_beta}/{cy3b_gamma}")
    fig.suptitle(param_text + "\nContrast shared across samples per channel; dots = all z-planes projected", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        name = (f"ParamTest_pre{int(preprocess)}_A647thr{a647_threshold}_Cy3Bthr{cy3b_threshold}"
                f"_dec{int(decompose)}_{time.strftime('%Y%m%d_%H%M%S')}.png")
        fig.savefig(os.path.join(output_dir, name), dpi=100)
        print(f"Saved figure: {os.path.join(output_dir, name)}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return results
