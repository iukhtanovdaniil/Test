import numpy as np
import pandas as pd
from skimage import io
from scipy import spatial
import os
import sys
from scipy.fft import fft2, fftshift, ifft2, ifftshift, fft, ifft, fftfreq
import cv2                                                     # for filtering vector fields
from skimage.morphology import disk        # morphology operations
import numpy.matlib as matlib
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import os
sys.path.append('/content/AFT-Alignment_by_Fourier_Transform/Python_implementation')
import AFT_tools as AFT

def AFT_AI_metric(im_list, df_spots, window_size, overlap,
                  im_mask=None, intensity_thresh=0, eccentricity_thresh=0,
                  single_frame=False):
    """
    Calculate alignment of tracks with local fiber orientation using AFT.
    
    Parameters:
    -----------
    im_list : list of str
        List of file paths to images.
    df_spots : pd.DataFrame
        DataFrame containing spot positions and tracks.
    window_size : int
        Window size for AFT analysis.
    overlap : float
        Overlap between windows
    im_mask : np.array, optional
        Mask for AFT analysis (default: None).
    intensity_thresh : float, optional
        Intensity threshold for AFT (default: 0).
    eccentricity_thresh : float, optional
        Eccentricity threshold for AFT (default: 0).
    single_frame : bool, optional
        Whether images are single-frame (default: False).
    
    Returns:
    --------
    df_subset_out : pd.DataFrame
        DataFrame with track-AFT alignment metrics for all images.
    """
    # Make the im list current - extract the name of the image that corresponds with image names in spot table
    im_list_current = [] 

    #take basename of each image in image folder
    for im_file in im_list: 
        filename = os.path.basename(im_file) 
        label = filename.replace('.tif', '') 

        im_list_current.append(label)

    im_list_current = np.array(im_list_current)

    df_subset_out = pd.DataFrame()

    for position in range(len(im_list_current)):
        # Load current image
        im = io.imread(im_list[position])
        if im.ndim == 3 and single_frame:
            im = im[0,] #take only first frame

        # Run AFT on image
        x, y, u, v, im_theta, im_eccentricity = AFT.image_local_order(
            im, window_size, overlap,
            im_mask, intensity_thresh, eccentricity_thresh,
            plot_overlay=False, plot_angles=False, plot_eccentricity=False,
            save_figures=False, save_path=''
        )

        # Prepare AFT coordinates
        AFT_coords = np.column_stack((x, y))
        ECM_labels = [f"ECM_{i+1}" for i in range(len(AFT_coords))] #give a unique ID for each ECM

        # Get subset of tracks for this image
        df_subset = df_spots.loc[df_spots.File_name_raw == im_list_current[position]]

        df_track_out = pd.DataFrame(columns=[
            'TRACK_ID','FRAME','track_angle','AFT_angle',
            'track_angle_u','track_angle_v','AFT_angle_u','AFT_angle_v'
        ])
        ###### maybe process each track not through the cycle and addition to the empty list but through the pandas df?
        # Process each track
        for trackID in df_subset.TRACK_ID.unique():
            df_track = df_subset.loc[df_subset.TRACK_ID == trackID].sort_values('FRAME').drop_duplicates(subset=['FRAME'], ignore_index=True) # take part of the Spot table with unique Trac_ID

            AFT_angle, track_angle, current_time_point, AFT_ecc, ECM_nearest, ECM_x, ECM_y = [], [], [], [], [], [], [] #create empty lists for each feature

            for time_point in range(len(df_track)-1):
                ######## maybe do start and finnish point through the shift(-1)? 
                track_current = [df_track.loc[time_point, 'POSITION_X'], df_track.loc[time_point, 'POSITION_Y']] #extract start point 
                track_next = [df_track.loc[time_point+1, 'POSITION_X'], df_track.loc[time_point+1, 'POSITION_Y']] #extract finish point 

                d_closest, idx_closest = spatial.KDTree(AFT_coords).query(track_current) #use KDTree to define closest point in AFT_coords
                closest_ECM_ID = ECM_labels[idx_closest] #define closest ECM_ID

                #append to the lists with variables 
                ECM_nearest.append(closest_ECM_ID)
                ECM_x.append(np.array(AFT_coords[idx_closest])[0])
                ECM_y.append(np.array(AFT_coords[idx_closest])[1])

                # Get AFT angle and eccentristy 
                if isinstance(im_theta, np.ndarray): #check that it comes in np.ndarray format
                    if im_theta.ndim == 2:  # single frame
                        AFT_angle.append(np.ravel(im_theta)[idx_closest]) #in 1D array
                        AFT_ecc.append(np.ravel(im_eccentricity)[idx_closest])
                    else:  # multi-frame
                        AFT_angle.append(np.ravel(im_theta[time_point])[idx_closest])
                        AFT_ecc.append(np.ravel(im_eccentricity[time_point])[idx_closest])
                elif isinstance(im_theta, list): #why did I write this block???? 
                    frame_theta = im_theta[time_point]
                    frame_ecc = im_eccentricity[time_point]
                    AFT_angle.append(np.ravel(frame_theta)[idx_closest])
                    AFT_ecc.append(np.ravel(frame_ecc)[idx_closest])


                # Track angle
                dx, dy = track_next[0]-track_current[0], track_next[1]-track_current[1]
                norm = np.sqrt(dx**2 + dy**2)
                track_angle.append(np.arctan2(dy/norm, dx/norm))
                current_time_point.append(df_track.loc[time_point, 'FRAME'])

            # Calculate alignment metrics
            AFT_angle = np.array(AFT_angle)
            AFT_ecc = np.array(AFT_ecc)
            track_angle = np.array(track_angle)
            current_time_point = np.array(current_time_point)
            ECM_nearest = np.array(ECM_nearest)
            
            AFT_track_angle = AFT_angle - track_angle
            AFT_track_angle_cos2 = np.cos(AFT_track_angle)**2

            # Build output DataFrame
            df_out = pd.DataFrame({
                'TRACK_ID': trackID,
                'FRAME': current_time_point,
                'image_name': im_list_current[position],
                'ECM_x': ECM_x,
                'ECM_y': ECM_y,
                'track_angle': track_angle,
                'AFT_angle': AFT_angle,
                'AFT-Δφ': AFT_track_angle,
                'AFT-AI': AFT_track_angle_cos2,
                'AFT_eccentricity': AFT_ecc,
                'track_angle_u': np.cos(track_angle),
                'track_angle_v': np.sin(track_angle),
                'AFT_angle_u': np.cos(AFT_angle),
                'AFT_angle_v': np.sin(AFT_angle)
            })

            df_track_for_merge = df_track.loc[df_track['FRAME'].isin(df_out['FRAME'])]
            df_out_merge = pd.merge(df_track_for_merge, df_out, on=['TRACK_ID','FRAME'], how='inner')
            
            df_track_out = pd.concat([df_track_out, df_out_merge], ignore_index=True)

        df_subset_out = pd.concat([df_subset_out, df_track_out], ignore_index=True)
        print(f"Image {position}: df_subset shape = {df_subset.shape}, frames = {df_subset['FRAME'].nunique()}")

    return df_subset_out

def AFT_order_parameter(im_list, window_size, overlap,
                        im_mask, intensity_thresh, eccentricity_thresh,
                        neighborhood_radius, single_frame, save_path=''):

    records = [] #create empty list for records

    #read image
    for im_file in im_list:
        im = io.imread(im_file)
        if im.ndim == 3 and single_frame:
            im = im[0,] #take only first frame if single frame 
        
        #extract filename as it is in the spot table
        filename = os.path.basename(im_file)
        label = filename.replace(".tif", "") 

        #Run AFT on image
        x, y, u, v, im_theta, im_ecc = AFT.image_local_order(im, window_size=window_size, overlap=overlap, im_mask=im_mask,
            intensity_thresh=intensity_thresh, eccentricity_thresh=eccentricity_thresh, plot_overlay=False, plot_angles=False,
            plot_eccentricity=False, save_figures=False, save_path=save_path
            )

        x = np.array(x) #extract x patch coords
        y = np.array(y) #extract y patch coords
        theta = np.array(im_theta) #extract theta 

        nrows, ncols = theta.shape #extract patch number as N patches in rows and N patches in columns

        for idx in range(len(x)): #go through the pathes indexes (go in x axis)
            r = idx // ncols #which column of the patch 
            c = idx % ncols #which row of the patch
            
            #NaN all pathes that are nearby the border and don't have a full neighboor 
            if r - neighborhood_radius < 0 or c - neighborhood_radius < 0 or r + neighborhood_radius >= nrows or c + neighborhood_radius >= ncols:
                order_val = np.nan
            #if have a full neighboorhood
            else:
                ref = theta[r, c] #take theta in patch centre 
                if np.isnan(ref): #if theta Nan order value is also NaN
                    order_val = np.nan
                else:
                    #take the neighboor theta values
                    patch = theta[r-neighborhood_radius:r+neighborhood_radius+1, c-neighborhood_radius:c+neighborhood_radius+1]
                    #calculate delta between neighbors and the central patch
                    delta = patch - ref 
                    #### copy calculation from AFT_tools.py, but for local ECM patch

                    #take the mean of cos(delta) as order value for the patch in the center of the neighborhood
                    order_array = np.cos(delta)**2 - 0.5
                    #take the mean of cos(delta) as order value for the patch in the center of the neighborhood
                    order_val = float(2 * np.nanmean(order_array))
                    

            records.append({
                "image_name": label,
                "ECM_ID": f"ECM_{idx+1}",
                "ECM_x": float(x[idx]),
                "ECM_y": float(y[idx]),
                "order_value": order_val,
                "window_size": window_size,
                "overlap": overlap,
                "neighborhood_radius": neighborhood_radius
            })

        print(f"Processed {filename}")

    return pd.DataFrame(records)

# transfer intensity calculation to segmentation pipeline?

'''
def intensity_calculation(im_list, window_size, overlap,
                        im_mask, intensity_thresh, eccentricity_thresh,
                        neighborhood_radius, single_frame, save_path=''):

    records = [] #create empty list for records

    #read image
    for im_file in im_list:
        im = io.imread(im_file)
        if im.ndim == 3 and single_frame:
            im = im[0,]

        #extract filename as it is in the spot table
        filename = os.path.basename(im_file)
        label = filename.replace(".tif", "")

        #Run AFT on image
        x, y, u, v, im_theta, im_ecc = AFT.image_local_order(im, window_size=window_size, overlap=overlap, im_mask=im_mask,
            intensity_thresh=intensity_thresh, eccentricity_thresh=eccentricity_thresh, plot_overlay=False, plot_angles=False,
            plot_eccentricity=False, save_figures=False, save_path=save_path
            )

        x_flat = np.ravel(x).astype(float) #extract x patch coords and make it 1D array
        y_flat = np.ravel(y).astype(float) #extract y patch coords and make it 1D array

        half = window_size // 2 #half of the patch size 
        H, W = im.shape #extract height and width of the image
        
        #take all overlap coordinates in x_flat and y_flat
        for xx, yy in zip(x_flat, y_flat):
            xx_i = int(xx) #take x coord
            yy_i = int(yy) #take y coord

            x1 = xx_i - half #left border
            x2 = xx_i + half #right border
            y1 = yy_i - half #top border
            y2 = yy_i + half #bottom border

            if x1 < 0 or y1 < 0 or x2 > W or y2 > H:
                mean_int = np.nan #NaN if patch is out of the image border
            else:
                patch = im[y1:y2, x1:x2] #take values in patch
                mean_int = float(np.mean(patch))

            if not np.isnan(mean_int) and mean_int < intensity_thresh:
                mean_int = np.nan

            records.append({
                'image_name': label,
                'ECM_x': xx_i,
                'ECM_y': yy_i,
                'intensity': mean_int,
            })

        print(f"Intensity processed: {filename}")

    return pd.DataFrame(records)
'''

### I don't know can I use this approach for density calculation. I am not sure, so, mb we should exclude it
'''
def density_calculation(im_list, window_size, overlap,
                        im_mask=None, intensity_thresh=0,
                        single_frame=True, save_path=''):

    records = []

    for im_file in im_list:
        im = io.imread(im_file)
        if im.ndim == 3 and single_frame:
            im = im[0,]

        filename = os.path.basename(im_file)
        label = filename.replace(".tif", "")

        x, y, u, v, im_theta, im_eccentricity = AFT.image_local_order(
            im,
            window_size=window_size,
            overlap=overlap,
            im_mask=im_mask,
            intensity_thresh=intensity_thresh,
            eccentricity_thresh=0,
            plot_overlay=False,
            plot_angles=False,
            plot_eccentricity=False,
            save_figures=False,
            save_path=''
        )

        x_flat = np.ravel(x).astype(float)
        y_flat = np.ravel(y).astype(float)

        half = window_size // 2
        H, W = im.shape

        for xx, yy in zip(x_flat, y_flat):
            xx_i = int(xx)
            yy_i = int(yy)

            x1 = xx_i - half
            x2 = xx_i + half
            y1 = yy_i - half
            y2 = yy_i + half

            if x1 < 0 or y1 < 0 or x2 > W or y2 > H:
                density_value = np.nan
            else:
                patch = im[y1:y2, x1:x2]
                fft2 = np.fft.fftshift(np.fft.fft2(patch))
                amp = np.abs(fft2)
                density_value = float(np.mean(amp))

            records.append({
                'image_name': label,
                'ECM_x': xx_i,
                'ECM_y': yy_i,
                'density': density_value,
            })

        print(f"Density processed: {filename}")

    return pd.DataFrame(records)
''' 
    
def data_saving_spots_only(
 df_subset_out,
 df_density,
 df_order,
 Results_Folder
 ):
 
 df = df_subset_out.copy()
 
 keys = ["image_name", "ECM_x", "ECM_y"]
 if df_order is not None:
    df = df.merge(df_order, on=keys, how="left")
 if df_density is not None:
    df = df.merge(df_density, on=keys, how="left")
 
 os.makedirs(Results_Folder, exist_ok=True)
 
 df.to_csv(os.path.join(Results_Folder, "merged_Spots_AFT.csv"), index=False)
 
 print("Successfully merged intensity, density, order into SPOTS table")
