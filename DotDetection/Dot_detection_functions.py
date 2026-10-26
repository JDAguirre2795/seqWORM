import os
import glob
import numpy as np
import tifffile

from HotPixel_functions import correct_hot_pixels, hot_pixel_file, load_hot_pixels

# Import big-fish components
try:
    from bigfish import stack
    from bigfish.detection import detect_spots, decompose_dense
    BIGFISH_AVAILABLE = True
except ImportError:
    BIGFISH_AVAILABLE = False
    print("Warning: big-fish not installed. Install with: pip install big-fish")


def load_channels(input_path):
    """
    Load the A647 and CY3B channels of a 3D multichannel stack as (Z, Y, X) arrays.
    Supports (Z, C, Y, X) stacks (e.g. from IMS to TIFF conversion) and (Z, Y, X, C) stacks.
    Returns None if the file is not a 3D multichannel stack.
    """
    with tifffile.TiffFile(input_path) as tif:
        axes = tif.series[0].axes
        img = tif.series[0].asarray()

    if axes == "ZCYX":
        return np.ascontiguousarray(img[:,0]), np.ascontiguousarray(img[:,1])
    elif img.ndim == 4:
        return np.ascontiguousarray(img[:,:,:,0]), np.ascontiguousarray(img[:,:,:,1])
    print(f"  Skipping {os.path.basename(input_path)}: axes {axes}, shape {img.shape} (expected a 3D multichannel stack)")
    return None


def detect_and_decompose(image, decompose_image, channel_name, threshold, voxel_size, spot_radius,
                         alpha, beta, gamma):
    """Detect spots on image, then decompose dense regions on decompose_image. Returns (detected, decomposed)."""
    print(f"  Processing {channel_name} channel...")

    spots, threshold = detect_spots(
        images=image,
        threshold=threshold,
        return_threshold=True,
        voxel_size=voxel_size,
        spot_radius=spot_radius)

    spots_post = spots
    if len(spots) > 0:
        try:
            spots_post, dense_regions, reference_spot = decompose_dense(
                image=decompose_image,
                spots=spots,
                voxel_size=voxel_size,
                spot_radius=spot_radius,
                alpha=alpha,
                beta=beta,
                gamma=gamma)
        except Exception as e:
            print(f"    {channel_name} - decomposition failed ({type(e).__name__}: {e}), keeping detected spots")

    print(f"    {channel_name} - before decomposition: {spots.shape}")
    print(f"    {channel_name} - after decomposition: {spots_post.shape}")
    return spots, spots_post


def process_tiff_file(input_path, output_path,
                     voxel_size, spot_radius,
                     a647_threshold, a647_alpha, a647_beta, a647_gamma,
                     cy3b_threshold, cy3b_alpha, cy3b_beta, cy3b_gamma,
                     decompose_path=None, z_trim=0):
    """
    Process a single TIFF file for FISH spot detection.

    Args:
        input_path: Path to input TIFF file (spots are detected on this image)
        output_path: Directory to save spot detection results
        voxel_size: Voxel size in nanometers (z, y, x)
        spot_radius: Spot radius in nanometers (z, y, x)
        a647_threshold: Threshold for A647 channel spot detection
        a647_alpha: Alpha parameter for A647 decomposition
        a647_beta: Beta parameter for A647 decomposition
        a647_gamma: Gamma parameter for A647 decomposition
        cy3b_threshold: Threshold for CY3B channel spot detection
        cy3b_alpha: Alpha parameter for CY3B decomposition
        cy3b_beta: Beta parameter for CY3B decomposition
        cy3b_gamma: Gamma parameter for CY3B decomposition
        decompose_path: Optional TIFF to run decomposition on (e.g. the raw image when detecting on a
            preprocessed one, so decomposition sees linear intensities). Defaults to input_path.
            If a <input_name>_hotpixels.csv exists next to input_path (written by hot-pixel-corrected
            preprocessing), the same hot pixels are corrected in the decomposition image.
        z_trim: Remove blank padding planes and then this many real planes at each end of the stack before
            detection. Saved spot z-coordinates stay in the original stack numbering.

    Returns:
        dict with spot counts, or None if the file was skipped
    """
    if not BIGFISH_AVAILABLE:
        raise ImportError("big-fish package is required. Install with: pip install big-fish")

    # Load the image(s)
    channels = load_channels(input_path)
    if channels is None:
        return None
    A647, CY3B = channels

    if decompose_path is not None:
        decompose_channels = load_channels(decompose_path)
        if decompose_channels is None or decompose_channels[0].shape != A647.shape:
            print(f"  Skipping {os.path.basename(input_path)}: decomposition image {decompose_path} missing or shape mismatch")
            return None
        A647_decompose, CY3B_decompose = decompose_channels
        print(f"  Decomposition on: {decompose_path}")
        hot_pixels = load_hot_pixels(hot_pixel_file(input_path))
        if hot_pixels is not None:
            A647_decompose = correct_hot_pixels(A647_decompose, hot_pixels)
            CY3B_decompose = correct_hot_pixels(CY3B_decompose, hot_pixels)
            print(f"  Hot pixels corrected in decomposition image: {len(hot_pixels)}")
    else:
        A647_decompose, CY3B_decompose = A647, CY3B

    # Remove blank padding planes, then z_trim real planes at each end
    z0 = 0
    if z_trim:
        reference = np.maximum(A647_decompose, CY3B_decompose)
        nonblank = np.where(reference.reshape(reference.shape[0], -1).max(axis=1) > 0)[0]
        z0, z1 = nonblank[0] + z_trim, nonblank[-1] - z_trim + 1
        if z1 - z0 < 3:
            print(f"  Skipping {os.path.basename(input_path)}: too few planes after trimming")
            return None
        A647, CY3B = A647[z0:z1], CY3B[z0:z1]
        A647_decompose, CY3B_decompose = A647_decompose[z0:z1], CY3B_decompose[z0:z1]
        print(f"  Planes used: {z0}-{z1 - 1} (padding removed + {z_trim} trimmed at each end)")

    spotsA647, spots_A647_post = detect_and_decompose(
        A647, A647_decompose, "A647", a647_threshold, voxel_size, spot_radius,
        a647_alpha, a647_beta, a647_gamma)

    spotsCY3B, spots_CY3B_post = detect_and_decompose(
        CY3B, CY3B_decompose, "CY3B", cy3b_threshold, voxel_size, spot_radius,
        cy3b_alpha, cy3b_beta, cy3b_gamma)

    # Back to original stack z-coordinates
    spotsA647, spots_A647_post, spotsCY3B, spots_CY3B_post = [
        s.copy() for s in (spotsA647, spots_A647_post, spotsCY3B, spots_CY3B_post)]  # may share memory
    for spots in (spotsA647, spots_A647_post, spotsCY3B, spots_CY3B_post):
        if len(spots):
            spots[:, 0] += z0

    # Extract the base name of the file
    base_name = os.path.basename(input_path).replace(".tif", "")

    # Get the pos folder name (e.g., pos0)
    pos_folder = os.path.basename(os.path.dirname(input_path))

    # Create subdirectory in the output path
    subdir = os.path.join(output_path, pos_folder)
    os.makedirs(subdir, exist_ok=True)

    # Save the results (decomposed spots, plus detected spots before decomposition)
    np.save(os.path.join(subdir, f"spots_A647_{base_name}.npy"), spots_A647_post)
    np.save(os.path.join(subdir, f"spots_CY3B_{base_name}.npy"), spots_CY3B_post)
    np.save(os.path.join(subdir, f"spots_A647_detected_{base_name}.npy"), spotsA647)
    np.save(os.path.join(subdir, f"spots_CY3B_detected_{base_name}.npy"), spotsCY3B)
    print(f"  ✓ Saved spots for: {base_name}")

    return {"folder": pos_folder, "file": base_name,
            "A647_detected": len(spotsA647), "A647_decomposed": len(spots_A647_post),
            "CY3B_detected": len(spotsCY3B), "CY3B_decomposed": len(spots_CY3B_post)}


def process_fish_spots(input_root, output_root=None,
                      voxel_size=(1000, 100.5, 100.5),
                      spot_radius=(1050, 300, 300),
                      a647_threshold=15, a647_alpha=0.5, a647_beta=1.7, a647_gamma=15,
                      cy3b_threshold=25, cy3b_alpha=0.5, cy3b_beta=1.7, cy3b_gamma=25,
                      decompose_root=None, z_trim=0):
    """
    Process FISH spot detection for ALL files in pos* folders (or in every subfolder if there are no pos* folders).
    Creates DetectedDots folder if output_root not specified.
    Saves spots_<channel>_<file>.npy (after decomposition), spots_<channel>_detected_<file>.npy (before
    decomposition) and a spot_counts.csv summary.

    Args:
        input_root: Root directory containing TIFF files (spots are detected on these)
        output_root: Root directory to save results (if None, creates DetectedDots)
        decompose_root: Optional root with the same folder/file structure to run decomposition on
            (e.g. raw TIFFs when input_root holds preprocessed TIFFs). Defaults to input_root.
        z_trim: Remove blank padding planes and then this many real planes at each end of each stack.
        voxel_size: Voxel size in nanometers (z, y, x)
        spot_radius: Spot radius in nanometers (z, y, x)
        a647_threshold: Threshold for A647 channel spot detection
        a647_alpha: Alpha parameter for A647 decomposition
        a647_beta: Beta parameter for A647 decomposition
        a647_gamma: Gamma parameter for A647 decomposition
        cy3b_threshold: Threshold for CY3B channel spot detection
        cy3b_alpha: Alpha parameter for CY3B decomposition
        cy3b_beta: Beta parameter for CY3B decomposition
        cy3b_gamma: Gamma parameter for CY3B decomposition
    """
    print("=" * 60)
    print("FISH Spot Detection Pipeline")
    print("=" * 60)
    
    print(f"Voxel size: {voxel_size}")
    print(f"Spot radius: {spot_radius}")
    print(f"A647 Parameters: threshold={a647_threshold}, alpha={a647_alpha}, beta={a647_beta}, gamma={a647_gamma}")
    print(f"CY3B Parameters: threshold={cy3b_threshold}, alpha={cy3b_alpha}, beta={cy3b_beta}, gamma={cy3b_gamma}")
    print(f"z trim: {z_trim}")
    
    if not BIGFISH_AVAILABLE:
        print("ERROR: big-fish package is not installed.")
        print("Install with: pip install big-fish")
        return
    
    # Set default output directory if not provided
    if output_root is None:
        output_root = os.path.join(input_root, "DetectedDots")
    
    # Create output directory
    os.makedirs(output_root, exist_ok=True)
    print(f"\nInput: {input_root}")
    print(f"Decomposition on: {decompose_root or input_root}")
    print(f"Output: {output_root}")
    
    # Find all pos* folders directly under input_root; if there are none, use every subfolder
    subfolders = [item for item in os.listdir(input_root)
                  if os.path.isdir(os.path.join(input_root, item))
                  and os.path.normpath(os.path.join(input_root, item)) != os.path.normpath(output_root)]
    pos_folders = [item for item in subfolders if item.startswith("pos")]
    if pos_folders:
        print(f"\nFound {len(pos_folders)} pos folders")
    else:
        pos_folders = subfolders
        print(f"\nNo pos folders found, using {len(pos_folders)} subfolders")
    
    # Process each pos folder
    counts = []
    for pos_folder in sorted(pos_folders):
        pos_path = os.path.join(input_root, pos_folder)
        
        # Look for ALL .tif files in this pos folder
        tiff_files = glob.glob(os.path.join(pos_path, "*.tif"))
        
        print(f"\nProcessing {pos_folder}: {len(tiff_files)} .tif files")
        
        # Process each file in this pos folder
        for i, tiff_file in enumerate(sorted(tiff_files), 1):
            filename = os.path.basename(tiff_file)
            print(f"\n[{i}/{len(tiff_files)}] Processing: {filename}")
            decompose_path = os.path.join(decompose_root, pos_folder, filename) if decompose_root else None
            result = process_tiff_file(
                tiff_file,
                output_root,
                decompose_path=decompose_path,
                z_trim=z_trim,
                voxel_size=voxel_size,
                spot_radius=spot_radius,
                a647_threshold=a647_threshold,
                a647_alpha=a647_alpha,
                a647_beta=a647_beta,
                a647_gamma=a647_gamma,
                cy3b_threshold=cy3b_threshold,
                cy3b_alpha=cy3b_alpha,
                cy3b_beta=cy3b_beta,
                cy3b_gamma=cy3b_gamma
            )
            if result:
                counts.append(result)

    # Save spot count summary
    if counts:
        import csv
        summary_path = os.path.join(output_root, "spot_counts.csv")
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(counts[0].keys()))
            writer.writeheader()
            writer.writerows(counts)
        print(f"\nSpot count summary: {summary_path}")

    print(f"\n" + "=" * 60)
    print(f"✓ Processing complete!")
    print(f"Results saved in: {output_root}")
    print("=" * 60)