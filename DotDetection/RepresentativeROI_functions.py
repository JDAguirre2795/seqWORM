"""
Spot density in a representative 3D box inside each manually outlined worm, plus DAPI + spot images.

Box: roi_xy x roi_xy pixels x roi_z planes (default 200 x 200 px x 10 planes = 20.1 x 20.1 x 10 um = 4040 um^3).
Selection per field:
    1. x-y: every position on a `stride`-px grid where the whole box lies inside the worm outline.
    2. z: for each x-y candidate, the roi_z consecutive analysed planes with the highest DAPI signal inside the box
       (the box sits in the middle of the worm's thickness).
    3. Representative box: the candidate whose spot density (A647 + CY3B detected) is closest to the median over all
       candidates. If all candidates have no spots (e.g. no-probe controls), the candidate closest to the worm centre.
Densities are reported per 1000 um^3 for the representative box, with the median / quartiles over all candidates.

Images per field and channel (A647, CY3B):
    <name>_<CH>_DAPI_spots.png: DAPI MIP + detected spots (all planes) + outline + box, and a zoom of the box
        (DAPI MIP of the box planes + spots inside the box).
    <name>_<CH>_DAPI_spots.tif: 2-channel ImageJ TIFF (DAPI MIP, detected spots drawn as 2-px discs).
"""
import os
import csv
import glob
import numpy as np
import tifffile
from scipy import ndimage as ndi

from HotPixel_functions import correct_hot_pixels, load_hot_pixels, hot_pixel_file


def _dapi_stack(raw_path, hot_pixels=None, z_trim=4):
    """DAPI (last channel) planes over the analysed range, hot pixels corrected. Returns stack and z0."""
    with tifffile.TiffFile(raw_path) as tif:
        s = tif.series[0]
        n_z, n_c = s.shape[:2]
        nonblank = [z for z in range(n_z) if s.pages[z * n_c].asarray().max() > 0]
        z0, z1 = nonblank[0] + z_trim, nonblank[-1] - z_trim + 1
        dapi = np.stack([s.pages[z * n_c + n_c - 1].asarray() for z in range(z0, z1)])
    if hot_pixels is not None:
        dapi = correct_hot_pixels(dapi, hot_pixels)
    return dapi, z0


def _box_sums(img2d, size):
    """Sum of img2d over every size x size box, indexed by the box's top-left corner."""
    c = np.pad(np.cumsum(np.cumsum(img2d.astype(np.float64), 0), 1), ((1, 0), (1, 0)))
    return c[size:, size:] - c[:-size, size:] - c[size:, :-size] + c[:-size, :-size]


def representative_roi(mask, dapi, z0, spots, roi_xy=200, roi_z=10, stride=25):
    """
    Choose the representative box (see module docstring).
    spots: {"A647": (N,3) z,y,x array, "CY3B": ...} detected spots in original stack z.
    Returns (box dict, list of all candidate dicts).
    """
    n_planes = dapi.shape[0]
    roi_z = min(roi_z, n_planes)
    inside = _box_sums(mask, roi_xy) >= roi_xy * roi_xy  # box fully inside the outline
    ys, xs = np.nonzero(inside)
    keep = (ys % stride == 0) & (xs % stride == 0)
    ys, xs = ys[keep], xs[keep]
    if len(ys) == 0:
        return None, []
    # DAPI per plane inside each candidate box -> best roi_z window
    plane_sums = np.stack([_box_sums(p, roi_xy)[ys, xs] for p in dapi.astype(np.float32)])  # (planes, candidates)
    win = np.cumsum(np.pad(plane_sums, ((1, 0), (0, 0))), 0)
    win = win[roi_z:] - win[:-roi_z]  # (n_windows, candidates)
    zstart = z0 + np.argmax(win, axis=0)
    cands = []
    for y, x, zs in zip(ys, xs, zstart):
        c = {"y0": int(y), "x0": int(x), "z0": int(zs), "z1": int(zs + roi_z)}
        for ch, s in spots.items():
            if len(s):
                sel = (s[:, 0] >= zs) & (s[:, 0] < zs + roi_z) & (s[:, 1] >= y) & (s[:, 1] < y + roi_xy) & \
                      (s[:, 2] >= x) & (s[:, 2] < x + roi_xy)
                c[ch] = int(sel.sum())
            else:
                c[ch] = 0
        c["total"] = sum(c[ch] for ch in spots)
        cands.append(c)
    totals = np.array([c["total"] for c in cands], float)
    if totals.max() > 0:
        best = int(np.argmin(np.abs(totals - np.median(totals))))
        rule = "closest to median density"
    else:
        cy, cx = np.argwhere(mask).mean(0)
        d = [(c["y0"] + roi_xy / 2 - cy) ** 2 + (c["x0"] + roi_xy / 2 - cx) ** 2 for c in cands]
        best = int(np.argmin(d))
        rule = "no spots in any box: closest to worm centre"
    box = dict(cands[best], rule=rule)
    return box, cands


def _count(s, box, roi_xy):
    if len(s) == 0:
        return 0, s
    sel = (s[:, 0] >= box["z0"]) & (s[:, 0] < box["z1"]) & (s[:, 1] >= box["y0"]) & (s[:, 1] < box["y0"] + roi_xy) & \
          (s[:, 2] >= box["x0"]) & (s[:, 2] < box["x0"] + roi_xy)
    return int(sel.sum()), s[sel]


def _dot_image(shape, s, radius=2):
    img = np.zeros(shape, dtype=np.uint8)
    if len(s):
        img[s[:, 1], s[:, 2]] = 255
        img = (ndi.binary_dilation(img > 0, structure=np.ones((2 * radius + 1, 2 * radius + 1))) * 255).astype(np.uint8)
    return img


def _save_images(name, ch, dapi_mip, dapi_box_mip, mask, box, s_all, s_box, roi_xy, out_dir, color):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.measure import find_contours

    # 2-channel ImageJ TIFF: DAPI MIP + detected spots (all planes) as discs
    tifffile.imwrite(os.path.join(out_dir, f"{name}_{ch}_DAPI_spots.tif"),
                     np.stack([dapi_mip, _dot_image(dapi_mip.shape, s_all).astype(np.uint16) * 257]),
                     imagej=True, metadata={"axes": "CYX", "Labels": ["DAPI MIP", f"{ch} detected spots"], "mode": "composite"})

    fig, axs = plt.subplots(1, 2, figsize=(17, 8.6), gridspec_kw={"width_ratios": [1, 1]})
    lo, hi = np.percentile(dapi_mip, [1, 99.7])
    ax = axs[0]
    ax.imshow(dapi_mip, cmap="gray", vmin=lo, vmax=max(hi, lo + 1))
    if len(s_all):
        ax.scatter(s_all[:, 2], s_all[:, 1], s=1.2, c=color, linewidths=0)
    for c in find_contours(mask.astype(float), 0.5):
        ax.plot(c[:, 1], c[:, 0], c="#ffd60a", lw=1.2)
    if box:
        ax.add_patch(plt.Rectangle((box["x0"], box["y0"]), roi_xy, roi_xy, fill=False, ec="#00e5ff", lw=1.8))
    ax.set_title(f"{name} | DAPI MIP + {ch} detected spots (all planes: {len(s_all)})\n"
                 f"yellow = worm outline, cyan = representative box", fontsize=10)
    ax = axs[1]
    if box:
        lo2, hi2 = np.percentile(dapi_box_mip, [1, 99.7])
        ax.imshow(dapi_box_mip, cmap="gray", vmin=lo2, vmax=max(hi2, lo2 + 1))
        if len(s_box):
            ax.scatter(s_box[:, 2] - box["x0"], s_box[:, 1] - box["y0"], s=40, facecolors="none", edgecolors=color, linewidths=1.1)
        ax.set_title(f"Representative box: {roi_xy}x{roi_xy} px x {box['z1'] - box['z0']} planes (z {box['z0']}-{box['z1'] - 1})\n"
                     f"DAPI MIP of box planes + {len(s_box)} {ch} spots inside the box", fontsize=10)
    else:
        ax.text(0.5, 0.5, "No box fits inside the outline", ha="center", va="center", transform=ax.transAxes)
    for a in axs:
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(out_dir, f"{name}_{ch}_DAPI_spots.png"), dpi=90, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)

    # Stand-alone zoom of the representative box (PNG + 2-channel TIFF)
    if box:
        fig, ax = plt.subplots(figsize=(8, 8.4))
        lo2, hi2 = np.percentile(dapi_box_mip, [1, 99.7])
        ax.imshow(dapi_box_mip, cmap="gray", vmin=lo2, vmax=max(hi2, lo2 + 1))
        if len(s_box):
            ax.scatter(s_box[:, 2] - box["x0"], s_box[:, 1] - box["y0"], s=60, facecolors="none", edgecolors=color, linewidths=1.3)
        ax.plot([10, 10 + 5 / 0.1005], [roi_xy - 10, roi_xy - 10], c="white", lw=3)  # 5 um scale bar
        ax.text(10, roi_xy - 16, "5 µm", color="white", fontsize=10)
        ax.set_title(f"{name} | {ch} | representative box {roi_xy}x{roi_xy} px x {box['z1'] - box['z0']} planes "
                     f"(z {box['z0']}-{box['z1'] - 1})\nDAPI MIP of box planes + {len(s_box)} detected spots", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{name}_{ch}_box_zoom.png"), dpi=110, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)
        local = s_box.copy()
        if len(local):
            local[:, 1] -= box["y0"]
            local[:, 2] -= box["x0"]
        tifffile.imwrite(os.path.join(out_dir, f"{name}_{ch}_box_zoom.tif"),
                         np.stack([dapi_box_mip, _dot_image(dapi_box_mip.shape, local).astype(np.uint16) * 257]),
                         imagej=True, metadata={"axes": "CYX", "Labels": ["DAPI MIP (box planes)", f"{ch} spots in box"], "mode": "composite"})


def roi_density_folders(raw_root, pre_root, dots_root, inputs_root, outlines_root, out_root,
                        roi_xy=200, roi_z=10, stride=25, pixel_um=0.1005, z_step_um=1.0, z_trim=4, only=None):
    """Representative-box densities and DAPI + spot images for every outlined field -> roi_density_counts.csv."""
    vol_um3 = (roi_xy * pixel_um) ** 2 * roi_z * z_step_um
    rows = []
    for mip_path in sorted(glob.glob(os.path.join(inputs_root, "*", "*_mips.tif"))):
        rel = os.path.relpath(mip_path, inputs_root)[:-len("_mips.tif")]
        folder, name = os.path.dirname(rel), os.path.basename(rel)
        if only and name not in only:
            continue
        row = {"folder": folder, "file": name}
        mask_p = os.path.join(outlines_root, rel + "_outline.tif")
        hot = load_hot_pixels(hot_pixel_file(os.path.join(pre_root, folder, name + ".tif")))
        dapi, z0 = _dapi_stack(os.path.join(raw_root, folder, name + ".tif"), hot, z_trim)
        spots = {ch: np.load(os.path.join(dots_root, folder, f"spots_{ch}_detected_{name}.npy")) for ch in ("A647", "CY3B")}
        dec = {ch: np.load(os.path.join(dots_root, folder, f"spots_{ch}_{name}.npy")) for ch in ("A647", "CY3B")}
        if os.path.exists(mask_p):
            mask = tifffile.imread(mask_p) > 0
            box, cands = representative_roi(mask, dapi, z0, spots, roi_xy, roi_z, stride)
            row.update({"status": "ok" if box else "no box fits", "n_candidate_boxes": len(cands), "box_volume_um3": round(vol_um3, 1)})
        else:  # still make the full-field DAPI + spot images, without outline or box
            mask, box, cands = np.zeros(dapi.shape[1:], bool), None, []
            row["status"] = "no usable worm" if os.path.exists(os.path.join(outlines_root, rel + "_noworm.txt")) else "no outline"
        out_dir = os.path.join(out_root, folder)
        os.makedirs(out_dir, exist_ok=True)
        dapi_mip = dapi.max(0)
        dapi_box_mip = None
        if box:
            row.update({"box_y0": box["y0"], "box_x0": box["x0"], "box_z0": box["z0"], "box_z1": box["z1"] - 1, "box_rule": box["rule"]})
            dapi_box_mip = dapi[box["z0"] - z0:box["z1"] - z0, box["y0"]:box["y0"] + roi_xy, box["x0"]:box["x0"] + roi_xy].max(0)
            for ch in ("A647", "CY3B"):
                n_det, _ = _count(spots[ch], box, roi_xy)
                n_dec, _ = _count(dec[ch], box, roi_xy)
                all_d = np.array([c[ch] for c in cands], float) / vol_um3 * 1000
                row.update({f"{ch}_detected_in_box": n_det, f"{ch}_decomposed_in_box": n_dec,
                            f"{ch}_detected_per_1000um3": round(1000 * n_det / vol_um3, 2),
                            f"{ch}_decomposed_per_1000um3": round(1000 * n_dec / vol_um3, 2),
                            f"{ch}_detected_per_1000um3_median_all_boxes": round(float(np.median(all_d)), 2),
                            f"{ch}_detected_per_1000um3_q25_all_boxes": round(float(np.percentile(all_d, 25)), 2),
                            f"{ch}_detected_per_1000um3_q75_all_boxes": round(float(np.percentile(all_d, 75)), 2)})
        for ch, color in (("A647", "#ff2d95"), ("CY3B", "#30d158")):
            s_box = _count(spots[ch], box, roi_xy)[1] if box else np.zeros((0, 3), int)
            _save_images(name, ch, dapi_mip, dapi_box_mip, mask, box, spots[ch], s_box, roi_xy, out_dir, color)
        rows.append(row)
        print(f"{rel}: {len(cands)} candidate boxes | box z {row.get('box_z0')}-{row.get('box_z1')} | "
              f"A647 {row.get('A647_detected_in_box')} CY3B {row.get('CY3B_detected_in_box')} spots in box", flush=True)

    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    os.makedirs(out_root, exist_ok=True)
    out_csv = os.path.join(out_root, "roi_density_counts.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {out_csv}")
    return rows
