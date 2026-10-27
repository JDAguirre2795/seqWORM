"""
3D soma segmentation for single-channel (488/GFP) Imaris .ims Z-stacks.

Batch version of Notebooks/Soma_Segmentation_3D.ipynb, with two changes found on the full
data set: soma centres must be further apart in Z than in Y/X (Z blur stretches each soma,
and a 4 um separation split single somas into upper/lower halves), and the volume/solidity
filters judge each candidate's refined shape, not its halo-inflated detected shape.

    1. smooth + triangle threshold          (neuron signal: soma + processes)
    2. anisotropic morphological opening    (strips thin processes)
    3. per-component watershed              (splits touching somas)
    4. intensity filter                     (drops dim debris)
    5. boundary refinement                  (trims each candidate's blur halo)
    6. volume / solidity filter             (on the refined shapes)
    7. measurement                          (volume, equivalent diameter, axis lengths, ...)

Use from a notebook (see Notebooks/Soma_Segmentation_batch.ipynb):

    from Soma_Segmentation_functions import process_folder
    all_somas, run_log = process_folder(r"D:\\data\\GFP")

or from a terminal (tiffprocessing env):

    python NeuronAnnotation/Soma_Segmentation_functions.py D:\\data\\GFP --boundary-fraction 0.25
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import tifffile
from matplotlib.figure import Figure
from scipy import ndimage as ndi
from skimage.color import label2rgb
from skimage.filters import threshold_triangle
from skimage.measure import regionprops, regionprops_table
from skimage.segmentation import watershed


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

# None = derived from soma_radius_um / process_radius_um exactly as in the notebook
# (see resolve_params).
DEFAULT_PARAMS = {
    # --- File ---
    "channel": 0,              # single 488 channel
    "resolution_level": 0,     # 0 = full resolution
    # Voxel size is read from each file's own metadata (it varies between acquisitions,
    # e.g. Z from 0.80 to 0.975 um). Set this to (Z, Y, X) um to force one value instead.
    "voxel_size_zyx": None,
    # Files with fewer Z planes than this are skipped (single 2D snapshots can't be
    # measured in 3D).
    "min_z_planes": 5,

    # --- Expected object sizes (um) ---
    "soma_radius_um": 4.0,       # ~4 um soma radius
    "process_radius_um": 0.25,   # ~0.5 um process diameter (near-diffraction-limited)
    # Must satisfy process_radius_um < opening_radius_um < soma_radius_um.
    "opening_radius_um": None,   # default: (process_radius_um + soma_radius_um) / 2

    # --- Watershed peak separation (minimum distance between two soma centres) ---
    "min_peak_separation_um": None,    # in Y/X; default: soma_radius_um
    # In Z; default: 2 x the Y/X value. Z blur makes a soma look ~2x taller than wide, so
    # at the Y/X value one soma was often split into an upper and a lower half.
    "min_peak_separation_z_um": None,

    # --- Denoising ---
    "smoothing_sigma_um": 0.15,

    # --- Intensity filter (step 4, on the detected candidate) ---
    # Multiples of each image's own triangle threshold.
    "min_mean_intensity_factor": 1.3,
    # The candidate's peak (its 99.9th-percentile brightness). Neurons have a bright core;
    # out-of-focus tissue and pharynx background are evenly dim. On 30 hand-labelled
    # candidates (5 files, 5 strains) neurons peaked at 4.4-54x the threshold and
    # background at <= 3.1x, so 3.7 sits in the gap. Mean brightness overlapped (1.56 vs 1.81).
    "min_peak_intensity_factor": 3.7,

    # --- Boundary refinement (step 5) ---
    # Each candidate keeps the voxels brighter than this fraction of the way from the
    # threshold up to its own peak. 0 = no trimming, higher = tighter.
    "boundary_fraction": 0.25,
    "peak_percentile": 99.9,  # the soma's "peak": robust to a few hot voxels
    # A candidate whose trimmed core falls into several pieces at least this big is several
    # somas (e.g. two neurons side by side): it is divided between those cores and each
    # part trimmed with its own peak. Smaller pieces are halo specks and are dropped.
    # Hand-checked: a real second neuron left a 29 um3 piece; specks were <= 12 um3.
    "min_split_piece_um3": 20.0,

    # --- Size/shape filter (step 6, on the REFINED shape) ---
    # Hand-labelled neurons refined to 52-153 um3; bright specks to <= 24 um3.
    "min_soma_volume_um3": 30.0,
    "max_soma_volume_um3": 2000.0,
    # 0 = off. Neurons here are elongated with process stubs (refined solidity down to
    # 0.49), so a roundness test rejected real somas; the peak filter does the sorting.
    "min_solidity": 0.0,
}


def resolve_params(params=None):
    """DEFAULT_PARAMS overridden by ``params``, with the derived (None) values filled in."""
    unknown = set(params or {}) - set(DEFAULT_PARAMS)
    if unknown:
        raise ValueError(f"unknown parameter(s): {sorted(unknown)}")
    p = {**DEFAULT_PARAMS, **(params or {})}
    soma_r, proc_r = p["soma_radius_um"], p["process_radius_um"]
    if p["opening_radius_um"] is None:
        p["opening_radius_um"] = (proc_r + soma_r) / 2
    if p["min_peak_separation_um"] is None:
        p["min_peak_separation_um"] = soma_r
    if p["min_peak_separation_z_um"] is None:
        p["min_peak_separation_z_um"] = 2 * p["min_peak_separation_um"]
    if not proc_r < p["opening_radius_um"] < soma_r:
        raise ValueError("opening_radius_um must lie between process_radius_um and soma_radius_um")
    return p


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _sorted_group_keys(group, prefix):
    """List keys like 'Channel 0', 'Channel 1', ... sorted numerically."""
    keys = [k for k in group.keys() if k.startswith(prefix)]
    return sorted(keys, key=lambda k: int(k.rsplit(" ", 1)[-1]))


def _channel_dataset(f, channel=0, resolution_level=0, time_point=0):
    tp_group = f["DataSet"][f"ResolutionLevel {resolution_level}"][f"TimePoint {time_point}"]
    channel_keys = _sorted_group_keys(tp_group, "Channel")
    return tp_group[channel_keys[channel]]["Data"]


def read_ims_info(ims_path, channel=0, resolution_level=0):
    """(Z, Y, X) shape and voxel size in um of one channel, without reading the image.

    Voxel size = physical extent (DataSetInfo/Image ExtMin/ExtMax, um) / voxel count, so it
    is right for any resolution level -- same as get_voxel_size() in
    "IMS to TIFF conversion.ipynb".
    """
    with h5py.File(ims_path, "r") as f:
        img_attrs = f["DataSetInfo"]["Image"].attrs

        def _attr_to_float(name):
            raw = img_attrs[name]
            chars = [c.decode() if isinstance(c, bytes) else c for c in raw]
            return float("".join(chars))

        ext_min = [_attr_to_float(f"ExtMin{i}") for i in range(3)]  # (x, y, z), um
        ext_max = [_attr_to_float(f"ExtMax{i}") for i in range(3)]
        z, y, x = _channel_dataset(f, channel, resolution_level).shape

    voxel_size_zyx = (
        (ext_max[2] - ext_min[2]) / z,
        (ext_max[1] - ext_min[1]) / y,
        (ext_max[0] - ext_min[0]) / x,
    )
    return (z, y, x), voxel_size_zyx


def load_ims_channel(ims_path, channel=0, resolution_level=0, time_point=0):
    """Read a single channel of an .ims file as a (Z, Y, X) numpy array."""
    with h5py.File(ims_path, "r") as f:
        return _channel_dataset(f, channel, resolution_level, time_point)[:]


# ---------------------------------------------------------------------------
# Segmentation steps
# ---------------------------------------------------------------------------

def ellipsoid_footprint(radius_um, voxel_size_zyx):
    """Boolean structuring element: a sphere of the given physical radius (um), expressed in
    voxels. ``radius_um`` may also be (Z, Y, X) radii for an ellipsoid."""
    radii_um = np.broadcast_to(radius_um, 3)
    radii_vox = [max(1, int(round(r / vs))) for r, vs in zip(radii_um, voxel_size_zyx)]
    zz, yy, xx = np.mgrid[
        -radii_vox[0]:radii_vox[0] + 1,
        -radii_vox[1]:radii_vox[1] + 1,
        -radii_vox[2]:radii_vox[2] + 1,
    ]
    dist2 = (zz / radii_vox[0]) ** 2 + (yy / radii_vox[1]) ** 2 + (xx / radii_vox[2]) ** 2
    return dist2 <= 1


def smooth_and_threshold(volume, voxel_size_zyx, smoothing_sigma_um):
    """Step 1: mild Gaussian denoise + triangle threshold. Returns (smoothed, threshold, mask).

    Same filter as skimage's gaussian(..., preserve_range=True) in the notebook, computed
    in float32 to halve memory on a full 2048x2048 stack.
    """
    sigma_vox = [smoothing_sigma_um / vs for vs in voxel_size_zyx]
    smoothed = ndi.gaussian_filter(volume.astype(np.float32), sigma=sigma_vox, mode="nearest")
    threshold = float(threshold_triangle(smoothed))
    return smoothed, threshold, smoothed > threshold


def strip_processes(mask, voxel_size_zyx, opening_radius_um):
    """Step 2: anisotropic opening removes structures thinner than the opening radius.

    Gives exactly ndi.binary_opening(mask, structure=ellipsoid_footprint(...)), ~10x faster.
    The footprint is every offset d with sum((d_i / r_i)^2) <= 1 (r_i = its radii in
    voxels). Scaling axis i by s_i = prod(r) / r_i turns that into |d|^2 <= prod(r)^2 in
    whole numbers, so with distance transforms in that metric:
        erosion:  a voxel survives iff its nearest background voxel lies outside the footprint
        dilation: a voxel is set  iff its nearest eroded voxel lies inside the footprint
    Squared distances are integers, so comparing against prod(r)^2 + 0.5 is exact.
    """
    footprint = ellipsoid_footprint(opening_radius_um, voxel_size_zyx)
    radii = [(n - 1) // 2 for n in footprint.shape]
    scale = int(np.prod(radii))
    sampling = [scale // r for r in radii]
    limit = scale ** 2 + 0.5

    # binary_erosion treats everything outside the stack as background: pad with it
    padded = np.pad(mask, 1, constant_values=False)
    eroded = ndi.distance_transform_edt(padded, sampling=sampling) ** 2 > limit
    eroded = eroded[1:-1, 1:-1, 1:-1]
    if not eroded.any():
        return eroded
    return ndi.distance_transform_edt(~eroded, sampling=sampling) ** 2 < limit


def _maximum_filter_ellipsoid(image, footprint):
    """Exactly ndi.maximum_filter(image, footprint=footprint, mode="nearest") for a convex,
    symmetric footprint (an ellipsoid_footprint), much faster for large footprints.

    Each (dz, dy) row of the footprint is a contiguous run of half-width w, so the max over
    the footprint is the max, over rows, of a 1D running max of width 2w + 1 shifted by
    (dz, dy). "nearest" clamps each axis independently, so edge-padding in Z/Y and
    mode="nearest" along X reproduce it exactly.
    """
    rz, ry, _ = [(n - 1) // 2 for n in footprint.shape]
    nz, ny, _ = image.shape
    padded_cache = {}
    out = None
    for dz in range(-rz, rz + 1):
        for dy in range(-ry, ry + 1):
            row = footprint[dz + rz, dy + ry]
            if not row.any():
                continue
            w = int(row.sum()) // 2
            if w not in padded_cache:
                run_max = ndi.maximum_filter1d(image, size=2 * w + 1, axis=2, mode="nearest")
                padded_cache[w] = np.pad(run_max, ((rz, rz), (ry, ry), (0, 0)), mode="edge")
            shifted = padded_cache[w][rz + dz:rz + dz + nz, ry + dy:ry + dy + ny]
            out = shifted.copy() if out is None else np.maximum(out, shifted, out=out)
    return out


def _peak_local_max_in_mask(image, footprint, region):
    """peak_local_max(image, footprint=footprint, labels=region, exclude_border=False),
    returning the same set of peaks, with the maximum filter replaced by
    _maximum_filter_ellipsoid. ``region`` is one boolean region; follows skimage 0.26's
    peak.py step by step for this case (min_distance=1, no thresholds, no peak limit).
    """
    threshold = image.min()
    roi = ndi.find_objects(region.astype(int))[0]
    label_mask = region[roi]
    img_object = image[roi].copy()
    img_object[~label_mask] = np.finfo(img_object.dtype).min

    image_max = _maximum_filter_ellipsoid(img_object, footprint)
    peaks = img_object == image_max
    if np.all(peaks[label_mask]):
        # no peak for a trivial image; isolated pixels in the region are peaks
        peaks[:] = False
        peaks[np.logical_xor(label_mask, ndi.binary_opening(label_mask))] = True
    peaks &= img_object > threshold

    coords = np.transpose(np.nonzero(peaks))
    return coords + np.array([s.start for s in roi])


def split_touching(opened_mask, voxel_size_zyx, min_peak_separation_um, min_peak_separation_z_um=None):
    """Step 3: marker-controlled watershed on the distance transform, per connected component.

    Two soma centres must be at least min_peak_separation_um apart in Y/X and
    min_peak_separation_z_um in Z (default: same as Y/X). Cropping to each component's
    padded box keeps distance_transform_edt from needing a full-frame buffer; same result,
    a fraction of the memory and time.
    """
    sep_z = min_peak_separation_um if min_peak_separation_z_um is None else min_peak_separation_z_um
    peak_footprint = ellipsoid_footprint((sep_z, min_peak_separation_um, min_peak_separation_um), voxel_size_zyx)
    # pad each crop by at least the peak-separation radius so distance/peak values near the
    # component's edge are computed with the same background context they'd see uncropped
    pad_vox = [r + 2 for r in ((np.array(peak_footprint.shape) - 1) // 2)]

    component_labels, _ = ndi.label(opened_mask)
    labels_ws = np.zeros(opened_mask.shape, dtype=np.int32)
    next_label = 1

    for i, bbox in enumerate(ndi.find_objects(component_labels), start=1):
        if bbox is None:
            continue
        padded = tuple(
            slice(max(0, s.start - p), min(dim, s.stop + p))
            for s, p, dim in zip(bbox, pad_vox, opened_mask.shape)
        )
        local_mask = component_labels[padded] == i

        local_dist = ndi.distance_transform_edt(local_mask, sampling=voxel_size_zyx)
        local_peaks = _peak_local_max_in_mask(local_dist, peak_footprint, local_mask)

        local_maxima = np.zeros_like(local_dist, dtype=bool)
        local_maxima[tuple(local_peaks.T)] = True
        local_markers, n_local = ndi.label(local_maxima)

        local_labels = watershed(-local_dist, local_markers, mask=local_mask)

        dest = labels_ws[padded]
        for local_id in range(1, n_local + 1):
            dest[local_labels == local_id] = next_label
            next_label += 1
        labels_ws[padded] = dest

    return labels_ws


def _peak_intensity(region_mask, intensity):
    """regionprops extra property: a candidate's 99.9th-percentile brightness."""
    return np.percentile(intensity[region_mask], 99.9)


def filter_candidates(labels_ws, smoothed, threshold, voxel_size_zyx, p):
    """Steps 4-6: intensity filter on the detected candidates, boundary refinement of the
    ones that pass (splitting any that hold several somas), then volume and solidity
    filters on the refined shapes.

    The detected shape still carries the blur halo (2-4x the refined volume, and a lumpy
    outline that drags solidity down), so size and shape are judged after refinement.

    Returns (final_labels, refined_labels, candidates, candidate_labels): kept somas
    relabelled 1..N as detected and as refined, the candidate table with pass/fail columns
    (one row per refined object; a split candidate gets one row per part, ``split_from``
    naming the original), and the candidate label volume including those splits.
    """
    with warnings.catch_warnings():
        # a few-voxel speck has no 3D convex hull: skimage warns and reports solidity inf;
        # the warnings would bury a batch log
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        detected = pd.DataFrame(regionprops_table(
            labels_ws,
            intensity_image=smoothed,
            spacing=voxel_size_zyx,
            properties=("label", "area", "centroid", "mean_intensity"),
            extra_properties=(_peak_intensity,),
        )).rename(columns={"area": "detected_volume_um3", "_peak_intensity": "peak_intensity"})
        detected["mean_ok"] = detected["mean_intensity"] >= threshold * p["min_mean_intensity_factor"]
        detected["peak_ok"] = detected["peak_intensity"] >= threshold * p["min_peak_intensity_factor"]
        detected["intensity_ok"] = detected["mean_ok"] & detected["peak_ok"]

        # refine only the candidates bright enough to be somas
        lookup = np.zeros(int(labels_ws.max()) + 1, dtype=labels_ws.dtype)
        bright = detected.loc[detected["intensity_ok"], "label"].to_numpy()
        lookup[bright] = bright
        split_labels, refined_all, parent = refine_and_split(
            lookup[labels_ws], smoothed, threshold, p["boundary_fraction"], p["peak_percentile"],
            float(np.prod(voxel_size_zyx)), p["min_split_piece_um3"])
        # dim candidates keep their original label (shown as rejected in the QC figure)
        split_labels = np.where(split_labels > 0, split_labels, labels_ws)

        refined = pd.DataFrame(regionprops_table(
            refined_all, spacing=voxel_size_zyx, properties=("label", "area", "solidity", "centroid"),
        )).rename(columns={"area": "volume_um3"})

    # one row per refined object, carrying its original candidate's brightness tests
    refined["split_from"] = refined["label"].map(parent).fillna(0).astype(int)
    refined["source"] = np.where(refined["split_from"] > 0, refined["split_from"], refined["label"])
    det_cols = ["label", "mean_intensity", "peak_intensity", "mean_ok", "peak_ok", "intensity_ok"]
    refined = refined.merge(detected[det_cols].rename(columns={"label": "source"}), on="source")
    det_volume = np.bincount(split_labels.ravel()) * np.prod(voxel_size_zyx)
    refined["detected_volume_um3"] = det_volume[refined["label"].to_numpy()]
    dim = detected[~detected["label"].isin(refined["source"])].assign(split_from=0)
    candidates = pd.concat([refined.drop(columns="source"), dim], ignore_index=True)

    candidates["volume_ok"] = candidates["volume_um3"].between(p["min_soma_volume_um3"], p["max_soma_volume_um3"])
    candidates["solidity_ok"] = candidates["solidity"] >= p["min_solidity"]
    candidates["kept"] = (candidates["intensity_ok"].astype(bool) & candidates["volume_ok"]
                          & candidates["solidity_ok"])
    candidates = candidates.sort_values("label", ignore_index=True)

    new_ids = np.zeros(int(split_labels.max()) + 1, dtype=np.uint16)
    kept = candidates.loc[candidates["kept"], "label"].to_numpy()
    new_ids[kept] = np.arange(1, len(kept) + 1)
    candidates["soma_id"] = new_ids[candidates["label"].to_numpy()]
    candidates = candidates[[
        "label", "split_from", "soma_id", "kept", "intensity_ok", "mean_ok", "peak_ok", "volume_ok",
        "solidity_ok", "mean_intensity", "peak_intensity", "volume_um3", "solidity",
        "detected_volume_um3", "centroid-0", "centroid-1", "centroid-2",
    ]]
    return new_ids[split_labels], new_ids[refined_all], candidates, split_labels


def _trim_to_core(obj, local, threshold, boundary_fraction, peak_percentile):
    """One object's bright core: the voxels of ``obj`` brighter than ``boundary_fraction``
    of the way from the threshold to the object's own peak, largest piece only, holes
    (e.g. a dimmer nucleus) filled. Returns ``obj`` itself if nothing passes."""
    peak = np.percentile(local[obj], peak_percentile)
    cutoff = threshold + boundary_fraction * (peak - threshold)
    core_labels, n_pieces = ndi.label(obj & (local > cutoff))
    if n_pieces == 0:
        # nothing above the cutoff: keep the detected label rather than lose a detected soma
        return obj
    # keep the main body only (drops stray halo specks), then fill a dimmer interior back
    # in; per-plane too, since coarse Z rarely closes it in 3D
    sizes = np.bincount(core_labels.ravel())[1:]
    core = core_labels == (np.argmax(sizes) + 1)
    core = ndi.binary_fill_holes(core)
    for z in range(core.shape[0]):
        core[z] = ndi.binary_fill_holes(core[z])
    return core & obj


def refine_boundaries(final_labels, smoothed, threshold, boundary_fraction, peak_percentile=99.9):
    """Trim each labelled object to its bright core, same IDs as the input.

    Inside its detected label, an object keeps the voxels brighter than ``boundary_fraction``
    of the way from the threshold up to its own peak, so bright and dim somas are trimmed
    consistently. Never adds voxels and never drops an object.
    """
    refined_labels = np.zeros_like(final_labels)
    for soma_id, bbox in enumerate(ndi.find_objects(final_labels), start=1):
        if bbox is None:
            continue
        obj = final_labels[bbox] == soma_id
        core = _trim_to_core(obj, smoothed[bbox], threshold, boundary_fraction, peak_percentile)
        refined_labels[bbox][core] = soma_id
    return refined_labels


def refine_and_split(labels, smoothed, threshold, boundary_fraction, peak_percentile,
                     voxel_volume_um3, min_split_piece_um3):
    """Step 5: refine_boundaries, except that a candidate whose trimmed core falls into
    several pieces of at least ``min_split_piece_um3`` is treated as several somas.

    Its detected region is divided between those cores (watershed on brightness, so each
    voxel joins the core it is brightest towards), and each part is trimmed with its own
    peak -- so a dimmer neuron touching a brighter one is not trimmed by the brighter
    one's cutoff. The first part keeps the candidate's label; the others get new labels.

    Returns (split_labels, refined_labels, parent): the detected labels with splits
    applied, the refined labels, and {new label: original label} for the extra parts.
    """
    split_labels = labels.copy()
    refined_labels = np.zeros_like(labels)
    parent = {}
    next_id = int(labels.max()) + 1
    for lab, bbox in enumerate(ndi.find_objects(labels), start=1):
        if bbox is None:
            continue
        obj = labels[bbox] == lab
        local = smoothed[bbox]
        peak = np.percentile(local[obj], peak_percentile)
        cutoff = threshold + boundary_fraction * (peak - threshold)
        pieces, _ = ndi.label(obj & (local > cutoff))
        sizes = np.bincount(pieces.ravel())[1:] * voxel_volume_um3
        cores = np.flatnonzero(sizes >= min_split_piece_um3) + 1

        if len(cores) < 2:
            refined_labels[bbox][_trim_to_core(obj, local, threshold, boundary_fraction, peak_percentile)] = lab
            continue
        markers = np.zeros(obj.shape, dtype=np.int32)
        for i, core_id in enumerate(cores, start=1):
            markers[pieces == core_id] = i
        parts = watershed(-local, markers, mask=obj)
        for i in range(1, len(cores) + 1):
            part = parts == i
            new_lab = lab
            if i > 1:
                new_lab, next_id = next_id, next_id + 1
                parent[new_lab] = lab
                split_labels[bbox][part] = new_lab
            core = _trim_to_core(part, local, threshold, boundary_fraction, peak_percentile)
            refined_labels[bbox][core] = new_lab
    return split_labels, refined_labels, parent


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

SOMA_COLUMNS = [
    "soma_id", "volume_um3", "equivalent_diameter_um",
    "axis_major_um", "axis_middle_um", "axis_minor_um", "solidity",
    "centroid_z_um", "centroid_y_um", "centroid_x_um", "mean_intensity",
    "touches_border", "volume_before_um3", "volume_kept_pct",
]


def _touches_border(labels):
    """soma_id -> whether it touches any face of the stack (cut off, so measured too small)."""
    faces = [labels[0], labels[-1], labels[:, 0], labels[:, -1], labels[:, :, 0], labels[:, :, -1]]
    return set(np.unique(np.concatenate([f.ravel() for f in faces]))) - {0}


def measure_somas(refined_labels, final_labels, smoothed, voxel_size_zyx):
    """One row per soma: size, shape, position (um) and brightness of the refined labels.

    equivalent_diameter_um is the diameter of the sphere with the same volume, (6V/pi)^(1/3).
    The three axis lengths are those of the ellipsoid with the same second moments; treat
    the one along Z with care, since coarse Z sampling makes it the least precise.
    """
    if refined_labels.max() == 0:
        return pd.DataFrame(columns=SOMA_COLUMNS)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # see filter_candidates
        warnings.simplefilter("ignore", RuntimeWarning)
        props = pd.DataFrame(regionprops_table(
            refined_labels,
            intensity_image=smoothed,
            spacing=voxel_size_zyx,
            properties=("label", "area", "solidity", "centroid", "mean_intensity", "inertia_tensor_eigvals"),
        ))
    stats = pd.DataFrame({
        "soma_id": props["label"],
        "volume_um3": props["area"],
        "equivalent_diameter_um": (6 * props["area"] / np.pi) ** (1 / 3),
    })
    # ellipsoid axis lengths from the inertia tensor eigenvalues (as skimage's
    # axis_major_length does in 3D): L_i = sqrt(10 * (sum(ev) - 2 * ev_i))
    ev = props[[f"inertia_tensor_eigvals-{i}" for i in range(3)]].to_numpy()
    lengths = np.sqrt(np.clip(10 * (ev.sum(axis=1, keepdims=True) - 2 * ev), 0, None))
    lengths = -np.sort(-lengths, axis=1)
    stats["axis_major_um"], stats["axis_middle_um"], stats["axis_minor_um"] = lengths.T
    stats["solidity"] = props["solidity"]
    stats["centroid_z_um"] = props["centroid-0"]
    stats["centroid_y_um"] = props["centroid-1"]
    stats["centroid_x_um"] = props["centroid-2"]
    stats["mean_intensity"] = props["mean_intensity"]

    border = _touches_border(refined_labels)
    stats["touches_border"] = stats["soma_id"].isin(border)

    before = np.bincount(final_labels.ravel(), minlength=int(final_labels.max()) + 1) * np.prod(voxel_size_zyx)
    stats["volume_before_um3"] = before[stats["soma_id"].to_numpy()]
    stats["volume_kept_pct"] = 100 * stats["volume_um3"] / stats["volume_before_um3"]
    return stats[SOMA_COLUMNS]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def segment_volume(volume, voxel_size_zyx, params=None):
    """Run steps 1-5 on one (Z, Y, X) volume. Returns a dict of results."""
    p = resolve_params(params)
    smoothed, threshold, mask = smooth_and_threshold(volume, voxel_size_zyx, p["smoothing_sigma_um"])
    opened_mask = strip_processes(mask, voxel_size_zyx, p["opening_radius_um"])
    del mask
    labels_ws = split_touching(opened_mask, voxel_size_zyx, p["min_peak_separation_um"],
                               p["min_peak_separation_z_um"])
    del opened_mask
    final_labels, refined_labels, candidates, candidate_labels = filter_candidates(
        labels_ws, smoothed, threshold, voxel_size_zyx, p)
    del labels_ws
    somas = measure_somas(refined_labels, final_labels, smoothed, voxel_size_zyx)
    return {
        "threshold": threshold,
        "candidates": candidates,
        "candidate_labels": candidate_labels,
        "final_labels": final_labels,
        "refined_labels": refined_labels,
        "somas": somas,
        "mip": np.max(volume, axis=0),
    }


def _reject_reason(row, p):
    """Short reason a candidate was rejected, for the QC figure."""
    if row.get("excluded_by_review", False):
        return "excluded (review)"
    if not row["intensity_ok"]:
        return "dim" if not row["mean_ok"] else "no bright core"
    reasons = []
    if not row["volume_ok"]:
        reasons.append("small" if row["volume_um3"] < p["min_soma_volume_um3"] else "big")
    if not row["solidity_ok"]:
        reasons.append(f"shape {row['solidity']:.2f}")
    return ", ".join(reasons)


def _draw_labels(ax, image, detected, refined, rejected, reasons, fontsize, aspect=1.0):
    """One QC panel: grey image, kept detected (red) / refined (yellow) outlines with cyan
    soma IDs, rejected candidates (magenta) with their rejection reason."""
    ax.imshow(image, cmap="gray", vmin=0, vmax=image.max() or 1, aspect=aspect)
    if detected.any():
        ax.contour(detected > 0, levels=[0.5], colors="red", linewidths=1)
        ax.contour(refined > 0, levels=[0.5], colors="yellow", linewidths=1)
    if rejected.any():
        ax.contour(rejected > 0, levels=[0.5], colors="magenta", linewidths=0.8)
        for region in regionprops(rejected):
            y, x = region.centroid
            ax.text(x, y, f"{region.label}: {reasons.get(region.label, '')}", color="magenta",
                    fontsize=fontsize - 2, ha="center", va="center")
    for region in regionprops(refined):
        y, x = region.centroid
        ax.text(x, y, str(region.label), color="cyan", fontsize=fontsize, fontweight="bold",
                ha="center", va="center")
    ax.axis("off")


def save_qc_figure(mip, final_labels, refined_labels, path, title="", candidates=None,
                   candidate_labels=None, params=None, volume=None, voxel_size_zyx=None,
                   min_shown_um3=20.0, zoom_margin_px=120):
    """QC image of one stack, all max projections.

    Left: the whole field, kept somas coloured with their IDs. Middle: zoomed on the somas
    and rejected candidates. Right (when ``volume`` is given): the same zoom seen from the
    side (X across, Z down, true scale), which separates objects stacked in Z.
    Kept somas: detected outline red, refined yellow, ID cyan. Rejected candidates of at
    least ``min_shown_um3`` (detected volume): magenta, with the reason they were rejected.
    """
    p = resolve_params(params)
    side = volume is not None and voxel_size_zyx is not None
    fig = Figure(figsize=(24 if side else 16, 8))
    axes = fig.subplots(1, 3 if side else 2, gridspec_kw={"width_ratios": [1, 1, 1] if side else [1, 1]})
    norm = mip / mip.max() if mip.max() > 0 else mip
    refined_mip = np.max(refined_labels, axis=0)
    detected_mip = np.max(final_labels, axis=0)

    rejected_ids = np.zeros(1, dtype=np.int32)
    rejected = pd.DataFrame()
    if candidates is not None and candidate_labels is not None:
        rejected = candidates[~candidates["kept"] & (candidates["detected_volume_um3"] >= min_shown_um3)]
        rejected_ids = np.zeros(int(candidate_labels.max()) + 1, dtype=np.int32)
        rejected_ids[rejected["label"].to_numpy()] = rejected["label"].to_numpy()
    rejected_mip = (np.max(rejected_ids[candidate_labels], axis=0) if len(rejected)
                    else np.zeros(mip.shape, dtype=np.int32))
    reasons = {int(r["label"]): _reject_reason(r, p) for _, r in rejected.iterrows()}

    axes[0].imshow(label2rgb(refined_mip, image=norm, bg_label=0, alpha=0.45))
    for region in regionprops(refined_mip):
        y, x = region.centroid
        axes[0].text(x, y, str(region.label), color="white", fontsize=11, fontweight="bold",
                     ha="center", va="center")
    if rejected_mip.any():
        axes[0].contour(rejected_mip > 0, levels=[0.5], colors="magenta", linewidths=0.6)
    n_kept = len(np.unique(refined_labels[refined_labels > 0]))  # IDs can have gaps after exclusions
    axes[0].set_title(f"{title}: {n_kept} soma(s) kept")
    axes[0].axis("off")

    shown = (detected_mip > 0) | (rejected_mip > 0)
    if not shown.any():
        axes[1].imshow(norm, cmap="gray")
        axes[1].set_title("no candidates found")
        axes[1].axis("off")
        if side:
            axes[2].axis("off")
    else:
        ys, xs = np.nonzero(shown)
        y0, y1 = max(0, ys.min() - zoom_margin_px), min(mip.shape[0], ys.max() + zoom_margin_px + 1)
        x0, x1 = max(0, xs.min() - zoom_margin_px), min(mip.shape[1], xs.max() + zoom_margin_px + 1)
        _draw_labels(axes[1], norm[y0:y1, x0:x1], detected_mip[y0:y1, x0:x1],
                     refined_mip[y0:y1, x0:x1], rejected_mip[y0:y1, x0:x1], reasons, 12)
        axes[1].set_title("top view (zoom): kept = red/yellow + cyan ID; rejected = magenta: reason")

        if side:
            box = (slice(None), slice(y0, y1), slice(x0, x1))
            side_img = np.max(volume[box], axis=1).astype(np.float32)
            side_rejected = (np.max(rejected_ids[candidate_labels[box]], axis=1) if len(rejected)
                             else np.zeros(side_img.shape, dtype=np.int32))
            _draw_labels(axes[2], side_img, np.max(final_labels[box], axis=1),
                         np.max(refined_labels[box], axis=1), side_rejected, reasons, 12,
                         aspect=voxel_size_zyx[0] / voxel_size_zyx[2])
            axes[2].set_title("side view (zoom): X across, Z down (top of stack at top)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def _apply_soma_exclusions(result, file_name, out_dir):
    """Remove the somas of ``file_name`` listed in ``out_dir/excluded_somas.csv`` from a
    segment_volume result, in place: gone from both label volumes and the soma table, and
    marked in the candidate table (``excluded_by_review``) so the QC figure shows them as
    rejected. Remaining somas keep their IDs, so the exclusion list stays valid."""
    result["candidates"]["excluded_by_review"] = False
    excluded_csv = Path(out_dir) / "excluded_somas.csv"
    if not excluded_csv.exists():
        return
    excluded = pd.read_csv(excluded_csv)
    ids = excluded.loc[excluded["file"] == file_name, "soma_id"].astype(int).to_numpy()
    ids = ids[ids > 0]
    if not len(ids):
        return
    for key in ("final_labels", "refined_labels"):
        labels = result[key]
        labels[np.isin(labels, ids)] = 0
    result["somas"] = result["somas"][~result["somas"]["soma_id"].isin(ids)].reset_index(drop=True)
    candidates = result["candidates"]
    hit = candidates["soma_id"].isin(ids) & candidates["kept"]
    candidates.loc[hit, "excluded_by_review"] = True
    candidates.loc[hit, "kept"] = False


def process_file(ims_path, out_dir, params=None, save_candidates=True):
    """Segment one .ims file and write its outputs to ``out_dir``. Returns (somas, log_row)."""
    p = resolve_params(params)
    ims_path, out_dir = Path(ims_path), Path(out_dir)
    stem = ims_path.stem
    log = {"file": ims_path.name, "status": "ok", "n_somas": 0, "n_candidates": 0,
           "voxel_z_um": np.nan, "voxel_y_um": np.nan, "voxel_x_um": np.nan,
           "threshold": np.nan, "seconds": np.nan, "message": ""}
    t0 = time.time()

    shape, voxel_size_zyx = read_ims_info(ims_path, p["channel"], p["resolution_level"])
    if p["voxel_size_zyx"] is not None:
        voxel_size_zyx = tuple(p["voxel_size_zyx"])
    log["voxel_z_um"], log["voxel_y_um"], log["voxel_x_um"] = voxel_size_zyx
    if shape[0] < p["min_z_planes"]:
        log.update(status="skipped", message=f"only {shape[0]} Z plane(s); not a 3D stack")
        return pd.DataFrame(columns=["file"] + SOMA_COLUMNS), log

    volume = load_ims_channel(ims_path, p["channel"], p["resolution_level"])
    result = segment_volume(volume, voxel_size_zyx, p)
    _apply_soma_exclusions(result, ims_path.name, out_dir)

    somas = result["somas"].copy()
    somas.insert(0, "file", ims_path.name)
    for axis, vs in zip("zyx", voxel_size_zyx):
        somas[f"voxel_{axis}_um"] = vs

    out_dir.mkdir(parents=True, exist_ok=True)
    # ImageJ metadata carries the voxel size, so Fiji/napari open the labels at true scale
    tifffile.imwrite(
        out_dir / f"{stem}_soma_labels.tif", result["refined_labels"], imagej=True,
        resolution=(1 / voxel_size_zyx[2], 1 / voxel_size_zyx[1]),
        metadata={"spacing": voxel_size_zyx[0], "unit": "um", "axes": "ZYX"},
    )
    somas.to_csv(out_dir / f"{stem}_soma_stats.csv", index=False)
    if save_candidates:
        result["candidates"].to_csv(out_dir / f"{stem}_candidates.csv", index=False)
    save_qc_figure(result["mip"], result["final_labels"], result["refined_labels"],
                   out_dir / f"{stem}_qc.png", title=ims_path.name,
                   candidates=result["candidates"], candidate_labels=result["candidate_labels"], params=p,
                   volume=volume, voxel_size_zyx=voxel_size_zyx)
    del volume

    log.update(n_somas=len(somas), n_candidates=len(result["candidates"]),
               threshold=result["threshold"], seconds=round(time.time() - t0, 1))
    return somas, log


def drop_excluded_somas(somas, out_dir):
    """``somas`` without the rows listed in ``out_dir/excluded_somas.csv`` (file, soma_id)."""
    excluded_csv = Path(out_dir) / "excluded_somas.csv"
    if not excluded_csv.exists() or somas.empty:
        return somas
    excluded = pd.read_csv(excluded_csv)
    keys = set(zip(excluded["file"], excluded["soma_id"].astype(int)))
    drop = [(f, int(i)) in keys for f, i in zip(somas["file"], somas["soma_id"])]
    return somas[~np.array(drop, dtype=bool)].reset_index(drop=True)


def process_folder(folder, out_dir=None, params=None, pattern="*.ims", skip_existing=False,
                   save_candidates=True):
    """Segment every file matching ``pattern`` in ``folder``.

    Per file, writes to ``out_dir`` (default: <folder>/soma_segmentation):
        <name>_soma_labels.tif   refined label volume (uint16, voxel size in ImageJ metadata)
        <name>_soma_stats.csv    one row per soma
        <name>_candidates.csv    every candidate with its filter results (for tuning)
        <name>_qc.png            max projection with outlines, to check by eye
    and for the whole run:
        all_soma_stats.csv       every soma from every file
        run_log.csv              per file: ok / skipped / error, counts, voxel size, time
        params.json              the exact parameters used

    A file that fails is logged as an error and the run carries on. With
    ``skip_existing=True``, files that already have a stats CSV are read back instead of
    re-segmented (resume an interrupted run). Files listed in ``out_dir/excluded_files.csv``
    (a ``file`` column, e.g. stacks found by eye to hold no real somas) are never
    processed or counted, and somas listed in ``out_dir/excluded_somas.csv`` (``file`` and
    ``soma_id`` columns) are left out of ``all_soma_stats.csv``. soma_ids are only stable for
    unchanged parameters, so review that list after changing any. Returns (all_somas, run_log).
    """
    p = resolve_params(params)
    folder = Path(folder)
    out_dir = Path(out_dir) if out_dir is not None else folder / "soma_segmentation"
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(folder.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no files matching {pattern!r} in {folder}")
    (out_dir / "params.json").write_text(json.dumps(p, indent=2), encoding="utf-8")
    excluded_csv = out_dir / "excluded_files.csv"
    excluded = set(pd.read_csv(excluded_csv)["file"]) if excluded_csv.exists() else set()

    all_somas, logs = [], []
    for n, path in enumerate(files, start=1):
        prefix = f"[{n}/{len(files)}] {path.name}"
        existing = out_dir / f"{path.stem}_soma_stats.csv"
        if path.name in excluded:
            somas = pd.DataFrame()
            log = {"file": path.name, "status": "excluded", "n_somas": 0,
                   "message": f"listed in {excluded_csv.name}"}
            print(f"{prefix}: excluded")
        elif skip_existing and existing.exists():
            somas = pd.read_csv(existing)
            log = {"file": path.name, "status": "existing", "n_somas": len(somas)}
            print(f"{prefix}: already done ({len(somas)} soma(s))")
        else:
            try:
                somas, log = process_file(path, out_dir, p, save_candidates)
            except Exception as exc:  # one bad file must not stop a long run
                somas = pd.DataFrame()
                log = {"file": path.name, "status": "error", "n_somas": 0,
                       "message": f"{type(exc).__name__}: {exc}"}
            detail = log.get("message") or f"{log['n_somas']} soma(s) in {log.get('seconds')} s"
            print(f"{prefix}: {log['status']} - {detail}")
        logs.append(log)
        if len(somas):
            all_somas.append(somas)

    all_somas = pd.concat(all_somas, ignore_index=True) if all_somas else pd.DataFrame(columns=["file"] + SOMA_COLUMNS)
    all_somas = drop_excluded_somas(all_somas, out_dir)
    run_log = pd.DataFrame(logs)
    all_somas.to_csv(out_dir / "all_soma_stats.csv", index=False)
    run_log.to_csv(out_dir / "run_log.csv", index=False)

    counts = run_log["status"].value_counts().to_dict()
    print(f"\nDone: {len(all_somas)} soma(s) from {len(files)} file(s) {counts} -> {out_dir}")
    return all_somas, run_log


def main(argv=None):
    ap = argparse.ArgumentParser(description="Batch 3D soma segmentation of .ims files.")
    ap.add_argument("folder", help="folder containing .ims files")
    ap.add_argument("--out", default=None, help="output folder (default: <folder>/soma_segmentation)")
    ap.add_argument("--pattern", default="*.ims", help="file pattern, e.g. 'PS8292*.ims'")
    ap.add_argument("--skip-existing", action="store_true", help="resume: reuse files already processed")
    ap.add_argument("--boundary-fraction", type=float, default=None)
    ap.add_argument("--soma-radius-um", type=float, default=None)
    ap.add_argument("--min-solidity", type=float, default=None)
    ap.add_argument("--min-mean-intensity-factor", type=float, default=None)
    ap.add_argument("--min-peak-intensity-factor", type=float, default=None)
    args = ap.parse_args(argv)
    params = {k: v for k, v in {
        "boundary_fraction": args.boundary_fraction,
        "soma_radius_um": args.soma_radius_um,
        "min_solidity": args.min_solidity,
        "min_mean_intensity_factor": args.min_mean_intensity_factor,
        "min_peak_intensity_factor": args.min_peak_intensity_factor,
    }.items() if v is not None}
    process_folder(args.folder, args.out, params, args.pattern, args.skip_existing)


if __name__ == "__main__":
    main()
