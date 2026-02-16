import os
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
from skimage import io
import sys
sys.path.append('/content/AFT-Alignment_by_Fourier_Transform/Python_implementation')
import AFT_tools as AFT

def AFT_preview(im_list, Results_Folder,
                    window_size, overlap,
                    neighborhood_radius, eccentricity_thresh,
                    im_mask=None, intensity_thresh=None):
    """
    Run AFT preview analysis on the first image from im_list.

    Parameters
    ----------
    im_list : list
        List of image file paths.
    Results_Folder : str
        Path to the output results folder.
    window_size : int
        Window size for AFT.
    overlap : int
        Overlap between windows.
    neighborhood_radius : int
        Neighborhood radius for local order.
    eccentricity_thresh : float
        Eccentricity threshold for AFT.
    im_mask : ndarray, optional
        Binary mask image (same shape as input image).
    intensity_thresh : float, optional
        Intensity threshold for AFT.

    Returns
    -------
    out_path : str
        Path to saved preview PNG.
    """

    # --- Create output folder ---
    preview_dir = os.path.join(Results_Folder, "preview_images")
    os.makedirs(preview_dir, exist_ok=True)

    # --- Load and prepare preview image ---
    preview_image = io.imread(im_list[0])
    if preview_image.ndim > 2:
        preview_image = preview_image[0]

    print("Preview image shape:", preview_image.shape)

    # --- Print selected parameters ---
    print("\nSelected parameters:")
    print(f"  window_size = {window_size}")
    print(f"  overlap = {overlap}")
    print(f"  neighborhood_radius = {neighborhood_radius}")
    print(f"  eccentricity_thresh = {eccentricity_thresh}\n")

    # --- Run AFT ---
    x, y, u, v, im_theta, im_eccentricity = AFT.image_local_order(
        preview_image, window_size, overlap,
        im_mask, intensity_thresh, eccentricity_thresh,
        plot_overlay=False, plot_angles=False, plot_eccentricity=False,
        save_figures=False, save_path=''
    )

    # --- Build output filename (append variables) ---
    in_name = os.path.basename(im_list[0])
    stem, _ = os.path.splitext(in_name)

    out_name = (
        f"{stem}_ws{window_size}"
        f"_ov{overlap}"
        f"_nr{neighborhood_radius}"
        f"_et{eccentricity_thresh}.png"
    )
    out_path = os.path.join(preview_dir, out_name)

    # --- Plot with a side legend panel ---
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[4, 1])

    ax_img = fig.add_subplot(gs[0, 0])
    ax_leg = fig.add_subplot(gs[0, 1])

    ax_img.imshow(preview_image, cmap='gray')
    ax_img.quiver(
        x, y, u, v,
        color='yellow',
        pivot='mid',
        scale_units='xy',
        scale=overlap,
        headaxislength=0,
        headlength=0,
        width=0.005,
        alpha=0.6
    )
    ax_img.set_title(f"AFT Preview: {in_name}")
    ax_img.axis('off')

    ax_leg.axis('off')
    legend_text = (
        "Parameters\n"
        "-----------\n"
        f"window_size: {window_size}\n"
        f"overlap: {overlap}\n"
        f"neighborhood_radius: {neighborhood_radius}\n"
        f"eccentricity_thresh: {eccentricity_thresh}\n"
    )
    ax_leg.text(
        0.05, 0.95, legend_text,
        va='top', ha='left',
        family='monospace'
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved preview to: {out_path}")

    # --- Quick plot (inline) ---
    plt.figure(figsize=(8, 8))
    plt.imshow(preview_image, cmap='gray')
    plt.quiver(
        x, y, u, v,
        color='yellow',
        pivot='mid',
        scale_units='xy',
        scale=overlap,
        headaxislength=0,
        headlength=0,
        width=0.005,
        alpha=0.6
    )
    plt.title(f"AFT Preview: {in_name} (ws={window_size})")
    plt.axis('off')
    plt.show()

    return out_path

def plot_AFT_overlays(
    im_list,
    df_subset_out,
    window_size,
    overlap,
    im_mask,
    intensity_thresh,
    eccentricity_thresh,
    single_frame,
    px_size_x,
    px_size_y,
    output_image_folder
):
    """
    Plot and save overlays of AFT orientation field with tracks.

    Parameters
    ----------
    im_list : list of str
        Paths to images.
    im_list_current : list of str
        Names of images corresponding to df_subset_out.
    df_subset_out : pd.DataFrame
        DataFrame with track info and file names.
    window_size : int
        AFT window size.
    overlap : float
        AFT overlap.
    im_mask : np.array, optional
        Mask for AFT.
    intensity_thresh : float, optional
        AFT intensity threshold.
    eccentricity_thresh : float, optional
        AFT eccentricity threshold.
    single_frame : bool, optional
        Whether images are single-frame.
    px_size_x : float, optional
        Pixel size in x (for scaling positions).
    px_size_y : float, optional
        Pixel size in y (for scaling positions).
    output_image_folder : str, optional
        Folder where overlays will be saved.
    """
    im_list_current = []

    for im_file in im_list:
        filename = os.path.basename(im_file)
        label = filename.replace('.tif', '')

        im_list_current.append(label)

    im_list_current = np.array(im_list_current)

    cmap = plt.get_cmap('hsv')

    for position in range(len(im_list_current)):
        # Load current image
        im = io.imread(im_list[position])
        if single_frame:
            im = im[0,]

        # Run AFT
        x, y, u, v, im_theta, im_eccentricity = AFT.image_local_order(
            im, window_size, overlap,
            im_mask, intensity_thresh, eccentricity_thresh,
            plot_overlay=False, plot_angles=False, plot_eccentricity=False,
            save_figures=False, save_path=''
        )

        # Subset tracks for current image
        df_temp = df_subset_out.loc[df_subset_out.File_name_raw == im_list_current[position]].copy()
        df_temp['POSITION_X'] = df_temp['POSITION_X'] * px_size_x
        df_temp['POSITION_Y'] = df_temp['POSITION_Y'] * px_size_y

        # Create folder for current sequence
        sequence_folder = os.path.join(output_image_folder, df_temp.File_name_raw.unique()[0])
        os.makedirs(sequence_folder, exist_ok=True)

        # Loop over time points
        for time_point in range(len(df_temp.FRAME.unique()) - 1):
            plt.figure(frameon=False)

            if single_frame is False:
                plt.imshow(im[time_point,], cmap='gray', aspect='equal')
            else:
                plt.imshow(im, cmap='gray', aspect='equal')

            plt.gca().set_aspect('equal', adjustable='box')

            # Plot all tracks
            for track in df_temp.TRACK_ID.unique():
                plt.plot(
                    df_temp.loc[df_temp.TRACK_ID == track, 'POSITION_X'],
                    df_temp.loc[df_temp.TRACK_ID == track, 'POSITION_Y'],
                    linewidth=1, color='w'
                )

            # Plot AFT field
            if single_frame is False:
                plt.quiver(
                    x, y, u[time_point], v[time_point],
                    color='yellow', pivot='mid', scale_units='xy',
                    scale=overlap, headaxislength=0, headlength=0,
                    width=0.005, alpha=0.4
                )
            else:
                plt.quiver(
                    x, y, u, v,
                    color='yellow', pivot='mid', scale_units='xy',
                    scale=overlap, headaxislength=0, headlength=0,
                    width=0.005, alpha=0.4
                )

            # Plot current position marker
            plt.plot(
                df_temp.loc[time_point, 'POSITION_X'],
                df_temp.loc[time_point, 'POSITION_Y'],
                marker='o', linestyle='None', color=cmap(time_point * 2)
            )

            plt.axis('off')

            file_name = (
                f"{df_temp.Condition.unique()[0]}_"
                f"{df_temp.File_name_raw.unique()[0]}_"
                f"Overlay_AFT_single_frame_{single_frame}_"
                f"frame_{time_point}.png"
            )

            save_path = os.path.join(sequence_folder, file_name)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

