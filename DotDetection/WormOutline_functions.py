"""
Manual worm outlines (napari) and spot density per worm area.

1. prepare_outline_inputs: per stack, save a (3, Y, X) MIP tif (DAPI, A647, CY3B; analysed planes only, hot pixels
   corrected) and the 2D positions of the detected spots, so napari opens each field instantly.
2. annotate_outlines: napari session; draw the worm to analyse on a Labels layer with the polygon tool (several
   polygons add up; brush/eraser/fill for touch-ups). The mask is saved as <name>_outline.tif.
   [s] save + next | [n] no usable worm + next | [q] quit. Fields that already have an outline are skipped.
3. summarize_outlines: spots inside each outline (detected + decomposed, both channels), worm area (um^2) and
   spots per 1000 um^2 -> worm_outline_counts.csv.

Steps 1 and 3 only need numpy/tifffile; step 2 needs napari (wormfish_annotation environment).
"""
import os
import csv
import glob
import numpy as np
import tifffile

from HotPixel_functions import correct_hot_pixels, load_hot_pixels, hot_pixel_file

CHANNELS = ("DAPI", "A647", "CY3B")


def _mip_stack(raw_path, hot_pixels=None, z_trim=4):
    """(3, Y, X) MIPs of DAPI (last channel), A647 (0), CY3B (1) over the analysed planes."""
    with tifffile.TiffFile(raw_path) as tif:
        s = tif.series[0]
        n_z, n_c = s.shape[:2]
        nonblank = [z for z in range(n_z) if s.pages[z * n_c].asarray().max() > 0]
        z0, z1 = nonblank[0] + z_trim, nonblank[-1] - z_trim + 1
        idx = (n_c - 1, 0, 1)
        mips = np.zeros((3,) + s.shape[2:], dtype=np.uint16)
        for z in range(z0, z1):
            for k, c in enumerate(idx):
                np.maximum(mips[k], s.pages[z * n_c + c].asarray(), out=mips[k])
    if hot_pixels is not None:
        mips = correct_hot_pixels(mips, hot_pixels)
    return mips, (z0, z1)


def prepare_outline_inputs(raw_root, pre_root, dots_root, inputs_root, z_trim=4, overwrite=False):
    """Write <inputs_root>/<folder>/<name>_mips.tif and <name>_spots_<CH>.npy (2D y, x of detected spots)."""
    n = 0
    for folder in sorted(os.listdir(dots_root)):
        fdir = os.path.join(dots_root, folder)
        if not os.path.isdir(fdir):
            continue
        names = sorted(f[len("spots_A647_detected_"):-4] for f in os.listdir(fdir) if f.startswith("spots_A647_detected_"))
        out_dir = os.path.join(inputs_root, folder)
        os.makedirs(out_dir, exist_ok=True)
        for name in names:
            mip_path = os.path.join(out_dir, f"{name}_mips.tif")
            if os.path.exists(mip_path) and not overwrite:
                continue
            hot = load_hot_pixels(hot_pixel_file(os.path.join(pre_root, folder, name + ".tif")))
            mips, (z0, z1) = _mip_stack(os.path.join(raw_root, folder, name + ".tif"), hot, z_trim)
            tifffile.imwrite(mip_path, mips, imagej=True, metadata={"axes": "CYX", "Labels": list(CHANNELS)})
            for ch in ("A647", "CY3B"):
                spots = np.load(os.path.join(fdir, f"spots_{ch}_detected_{name}.npy"))
                np.save(os.path.join(out_dir, f"{name}_spots_{ch}.npy"), spots[:, 1:] if len(spots) else np.zeros((0, 2), int))
            n += 1
            print(f"  {folder} | {name}: planes {z0}-{z1 - 1}", flush=True)
    print(f"Prepared {n} fields in {inputs_root}")


def _fields(inputs_root):
    return sorted(glob.glob(os.path.join(inputs_root, "*", "*_mips.tif")))


def _outline_paths(mip_path, inputs_root, outlines_root):
    rel = os.path.relpath(mip_path, inputs_root)
    base = os.path.join(outlines_root, rel[:-len("_mips.tif")])
    return base + "_outline.tif", base + "_outline.csv", base + "_noworm.txt"


def check_outline_status(inputs_root, outlines_root):
    """Print which fields have an outline (✓), are marked no-worm (–) or are pending (□)."""
    done = noworm = 0
    lines = []
    for p in _fields(inputs_root):
        mask_p, _, nw_p = _outline_paths(p, inputs_root, outlines_root)
        if os.path.exists(mask_p):
            mark, done = "✓", done + 1
        elif os.path.exists(nw_p):
            mark, noworm = "–", noworm + 1
        else:
            mark = "□"
        lines.append(f"  {mark} {os.path.relpath(p, inputs_root)[:-len('_mips.tif')]}")
    total = len(lines)
    print(f"Fields: {total} | outlined: {done} | no usable worm: {noworm} | pending: {total - done - noworm}")
    print("\n".join(lines))


class WormOutlineAnnotator:
    """napari session: one field at a time, draw polygon(s) around the worm to analyse."""

    def __init__(self, inputs_root, outlines_root, redo=False, only=None):
        self.inputs_root, self.outlines_root = inputs_root, outlines_root
        fields = _fields(inputs_root)
        if only:  # specific fields by file name (e.g. ["ProtK_Male head"]); these are always (re)opened
            only = set(only)
            fields = [p for p in fields if os.path.basename(p)[:-len("_mips.tif")] in only]
            redo = True
        if not redo:
            fields = [p for p in fields if not any(os.path.exists(x) for x in (_outline_paths(p, inputs_root, outlines_root)[0],
                                                                              _outline_paths(p, inputs_root, outlines_root)[2]))]
        self.queue = fields
        self.viewer = None
        self.current = None

    def start(self):
        import napari
        if not self.queue:
            print("All fields already have an outline (or are marked no-worm).")
            return
        print(f"{len(self.queue)} fields to outline")
        self.viewer = napari.Viewer(title="Worm outlines")
        self._bind_keys()
        self._load_next()

    def _bind_keys(self):
        @self.viewer.bind_key("s", overwrite=True)
        def _save(_):
            if self._save():
                self._next_later()

        @self.viewer.bind_key("n", overwrite=True)
        def _noworm(_):
            self._mark_noworm()
            self._next_later()

        @self.viewer.bind_key("q", overwrite=True)
        def _quit(_):
            self._close_later()

    def _load_next(self):
        if not self.queue:
            print("All fields done!")
            from qtpy.QtCore import QTimer
            QTimer.singleShot(50, self.viewer.close)
            return
        self.current = self.queue.pop(0)
        name = os.path.basename(self.current)[:-len("_mips.tif")]
        folder = os.path.dirname(self.current)
        mips = tifffile.imread(self.current)
        v = self.viewer
        H, W = mips.shape[1:]
        start = np.zeros((H, W), dtype=np.uint8)
        mask_p, _, _ = _outline_paths(self.current, self.inputs_root, self.outlines_root)
        if os.path.exists(mask_p):  # re-load an existing outline when redoing
            start[tifffile.imread(mask_p) > 0] = 1
        pts = {ch: np.load(os.path.join(folder, f"{name}_spots_{ch}.npy")) for ch in ("A647", "CY3B")}

        # Layers are created once and only their data is swapped for each field: removing/re-adding layers can
        # make the OpenGL canvas draw freed buffers (access violation in paintGL) on some drivers.
        if "worm mask" not in v.layers:
            colors = {"DAPI": "gray", "A647": "magenta", "CY3B": "green"}
            for k, ch in enumerate(CHANNELS):
                v.add_image(mips[k], name=f"{ch} MIP", colormap=colors[ch], blending="additive", visible=(ch != "CY3B"))
            for ch, col in (("A647", "red"), ("CY3B", "cyan")):
                v.add_points(pts[ch], name=f"{ch} spots", size=6, face_color=col, border_width=0, opacity=0.6, visible=False)
            # Labels layer + polygon tool (fills directly into a mask; avoids the Shapes-layer triangulation bug)
            labels = v.add_labels(start, name="worm mask", opacity=0.35)
            self._bind_layer_keys(labels)
        else:
            for k, ch in enumerate(CHANNELS):
                v.layers[f"{ch} MIP"].data = mips[k]
            for ch in ("A647", "CY3B"):
                v.layers[f"{ch} spots"].data = pts[ch]
            v.layers["worm mask"].data = start
        for k, ch in enumerate(CHANNELS):
            lo, hi = np.percentile(mips[k], [1, 99.8])
            v.layers[f"{ch} MIP"].contrast_limits = (float(lo), float(max(hi, lo + 1)))
        labels = v.layers["worm mask"]
        labels.selected_label = 1
        v.layers.selection.active = labels
        labels.mode = "polygon"
        v.reset_view()
        print(f"  spots: A647 {len(pts['A647'])}, CY3B {len(pts['CY3B'])}")
        v.title = f"Worm outlines | {os.path.relpath(self.current, self.inputs_root)[:-len('_mips.tif')]} | {len(self.queue)} left"
        print(f"\nOutline: {os.path.relpath(self.current, self.inputs_root)[:-len('_mips.tif')]}  ({len(self.queue)} left after this)")
        print("  'worm mask' layer, polygon tool: click around the worm, double-click (or Enter) to fill the polygon.")
        print("  Several polygons add up; touch up with brush [p], eraser [e], fill [f]; back to polygon tool via the layer controls.")
        print("  Spot layers are hidden by default - tick them in the layer list to show them.")
        print("  [s] save + next | [n] no usable worm + next | [q] quit")

    def _bind_layer_keys(self, layer):
        # layer key bindings take precedence over viewer ones while the layer is selected
        layer.bind_key("s", lambda _: self._save() and self._next_later(), overwrite=True)
        layer.bind_key("n", lambda _: (self._mark_noworm(), self._next_later()), overwrite=True)
        layer.bind_key("q", lambda _: self._close_later(), overwrite=True)

    def _next_later(self):
        """Load the next field after the current key event has finished (swapping layers inside the key callback
        can crash the OpenGL canvas)."""
        from qtpy.QtCore import QTimer
        QTimer.singleShot(50, self._load_next)
        return True

    def _close_later(self):
        from qtpy.QtCore import QTimer
        print("Session closed. Run again to continue with the remaining fields.")
        QTimer.singleShot(50, self.viewer.close)

    def _save(self):
        mask = self.viewer.layers["worm mask"].data > 0
        if not mask.any():
            print("  Nothing drawn - draw the worm, or press [n] if there is no usable worm.")
            return False
        mask_p, csv_p, nw_p = _outline_paths(self.current, self.inputs_root, self.outlines_root)
        os.makedirs(os.path.dirname(mask_p), exist_ok=True)
        tifffile.imwrite(mask_p, mask.astype(np.uint8) * 255)
        if os.path.exists(nw_p):
            os.remove(nw_p)
        print(f"  ✓ Saved outline ({mask.sum() * 0.1005 ** 2:,.0f} um^2)")
        return True

    def _mark_noworm(self):
        mask_p, _, nw_p = _outline_paths(self.current, self.inputs_root, self.outlines_root)
        os.makedirs(os.path.dirname(nw_p), exist_ok=True)
        with open(nw_p, "w") as f:
            f.write("no usable worm in field\n")
        if os.path.exists(mask_p):  # replacing an earlier outline
            os.remove(mask_p)
        print("  – Marked as no usable worm")


def annotate_outlines(inputs_root, outlines_root, redo=False, new_process=True, only=None):
    """
    Open the napari outline session (blocks until the window is closed).
    only: list of file names to (re)do, e.g. ["ProtK_Male head"]; their saved outline is loaded for editing.
        Saving a field that was marked 'no usable worm' replaces that mark.
    new_process=True (default) runs napari in a separate Python process with the same environment, so the Qt event
    loop never runs inside the Jupyter kernel (which can crash the kernel on Windows). Messages are printed here.
    """
    if not new_process:
        import napari
        annotator = WormOutlineAnnotator(inputs_root, outlines_root, redo=redo, only=only)
        annotator.start()
        if annotator.viewer is not None:
            napari.run()
        return annotator

    import sys
    import subprocess
    env_dir = os.path.dirname(sys.executable)
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([env_dir, os.path.join(env_dir, "Library", "mingw-w64", "bin"),
                                   os.path.join(env_dir, "Library", "usr", "bin"), os.path.join(env_dir, "Library", "bin"),
                                   os.path.join(env_dir, "Scripts"), env.get("PATH", "")])
    env["PYTHONUNBUFFERED"] = "1"
    code = ("import sys; sys.path.insert(0, {mod!r}); import WormOutline_functions as W; "
            "W.annotate_outlines({i!r}, {o!r}, redo={r!r}, new_process=False, only={only!r})").format(
        mod=os.path.dirname(os.path.abspath(__file__)), i=inputs_root, o=outlines_root, r=redo,
        only=list(only) if only else None)
    proc = subprocess.Popen([sys.executable, "-c", code], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace")
    for line in proc.stdout:
        if "QStandardPaths" in line or "Warning" in line:
            continue
        print(line, end="", flush=True)
    proc.wait()
    if proc.returncode != 0:
        print(f"napari process exited with code {proc.returncode}")
    return None


def summarize_outlines(inputs_root, outlines_root, dots_root, out_csv=None, pixel_um=0.1005):
    """Spots inside each outline, worm area and densities -> CSV (also returns the rows)."""
    out_csv = out_csv or os.path.join(outlines_root, "worm_outline_counts.csv")
    rows = []
    for p in _fields(inputs_root):
        rel = os.path.relpath(p, inputs_root)
        folder, name = os.path.dirname(rel), os.path.basename(rel)[:-len("_mips.tif")]
        mask_p, _, nw_p = _outline_paths(p, inputs_root, outlines_root)
        row = {"folder": folder, "file": name}
        if os.path.exists(mask_p):
            mask = tifffile.imread(mask_p) > 0
            area = float(mask.sum()) * pixel_um ** 2
            row.update({"status": "outlined", "worm_area_um2": round(area, 1)})
            for ch in ("A647", "CY3B"):
                for kind, fname in (("detected", f"spots_{ch}_detected_{name}.npy"), ("decomposed", f"spots_{ch}_{name}.npy")):
                    s = np.load(os.path.join(dots_root, folder, fname))
                    inside = int(mask[s[:, 1], s[:, 2]].sum()) if len(s) else 0
                    row[f"{ch}_{kind}_total"] = int(len(s))
                    row[f"{ch}_{kind}_in_worm"] = inside
                    row[f"{ch}_{kind}_per_1000um2"] = round(1000 * inside / area, 2) if area else ""
        else:
            row["status"] = "no usable worm" if os.path.exists(nw_p) else "pending"
        rows.append(row)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    n = {s: sum(r["status"] == s for r in rows) for s in ("outlined", "no usable worm", "pending")}
    print(f"Saved {out_csv} | outlined {n['outlined']}, no usable worm {n['no usable worm']}, pending {n['pending']}")
    return rows
