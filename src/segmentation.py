import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 
import AFT_tools as AFT

from skimage import img_as_float, io
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.morphology import skeletonize
from skimage.measure import label

from skan import Skeleton, summarize
from skan.csr import skeleton_to_nx

from fractal import box_counting

# @title Apply contrast
    
# function calcSigma, calcLowerThresh, calcUpperThresh directly rewritten from TWOMBLI Java code 

#calculate sigma for the line width range
def calcSigma(lineWidth):
    return lineWidth / (2 * math.sqrt(3)) + 0.5

#calculate thresholds for the ridge detector
def calcLowerThresh(lineWidth, estimatedSigma, contrastLow, contrastHigh, darkline):
    clow = contrastLow
    if darkline:
        clow = 255 - contrastHigh

    value = abs(
        -2 * clow * (lineWidth / 2.0)
        / (math.sqrt(2 * math.pi) * estimatedSigma**3)
        * math.exp(-((lineWidth / 2.0)**2) / (2 * estimatedSigma**2))
    )
    return 0.17 * math.floor(value)

#calculate upper threshold 
def calcUpperThresh(lineWidth, estimatedSigma, contrastLow, contrastHigh, darkline):
    chigh = contrastHigh
    if darkline:
        chigh = 255 - contrastLow

    value = abs(
        -2 * chigh * (lineWidth / 2.0)
        / (math.sqrt(2 * math.pi) * estimatedSigma**3)
        * math.exp(-((lineWidth / 2.0)**2) / (2 * estimatedSigma**2))
    )
    return 0.17 * math.floor(value)

#enhance contrast function 
def enhance_contrast(image, intensity_clip_percent):
    img = img_as_float(image)
    #define low and high contrast levels
    low, high = np.percentile(
        img,
        (intensity_clip_percent / 2, 100 - intensity_clip_percent / 2)
    ) 

    if high <= low:
        return img
        print('Your high contrast level lower than low')
    #cut low and contrast levels and rescale to [0,1]
    img = np.clip(img, low, high)
    #normalized to [0,1]
    img = (img - low) / (high - low)

    #demonstrate results
    plt.figure(figsize=(5,5))
    plt.imshow(img, cmap='gray')
    plt.title("Enhanced contrast image")
    plt.axis('off')

    return img

'''
# remove_short_components (Maybe better to do that through the skan?)
def remove_short_components(mask, min_len):
    labeled = label(mask) #CCL - make mistakes, change to skan or mb I don't need to work with skeleton because I work with Ridge-detector mask???
    out = np.zeros_like(mask, dtype=bool) 

    for comp_id in range(1, labeled.max()+1): #go throug each comp_id
        comp = (labeled == comp_id) #extract commponent mask
        skel = skeletonize(comp) #skeletonize it
        length = skel.sum() #calculate the sum of skeleton pixels 

        if length >= min_len: #save or remove components based on their lenght 
            out[comp] = True 
        else:
            out[comp] = False

    return out
'''

def remove_short_components(skel, min_len):

    if skel is None or np.count_nonzero(skel) == 0:
        return skel
    
    # take summary table from skan
    sk = Skeleton(skel)
    summary = summarize(Skeleton(skel), separator='_')
    #print(summary.head()) #look at the summary table to understand what data we have and how to work with it
    #skel mask
    skel_out = np.zeros_like(skel, dtype=bool)
    skel_out_branch_filtered = np.zeros_like(skel, dtype=bool)
    #print((summary['branch_type'] == 2).value_counts())
    #print(summary.columns.tolist())
    # take bracnh_id and row for each branch in summary table
    
    #take only branches with endpoints or separate branches (not loops)
    endpoints_df = summary[(summary['branch_type'] == 0) | (summary['branch_type'] == 1)]
    endpoints_df = endpoints_df[endpoints_df['branch_distance'] >= min_len]
    circles_df = summary[summary['branch_type'] == 3]
    circles_df = circles_df[circles_df['branch_distance'] >= min_len]
    not_endpoints_df = summary[(summary['branch_type'] == 2)]

    concat = pd.concat([endpoints_df, not_endpoints_df, circles_df])
    for branch_id, row in concat.iterrows():
        coords = sk.path_coordinates(branch_id)
        for y, x in coords:
            skel_out[int(y), int(x)] = True
    '''
        #check if branch length is higher than min_len, if yes - save it in the output skeleton mask, if not - skip it
        if row["branch_distance"] >= min_len:

            coords = sk.path_coordinates(branch_id)
            for y, x in coords:
                skel_out_endpoint_filtered[int(y), int(x)] = True
    if row['branch_type'] == 2 or row['branch_type'] == 3:
        coords = sk.path_coordinates(branch_id)
        for y, x in coords:
            skel_out_branch_filtered[int(y), int(x)] = True
    '''
    
    return skel_out

#hessian fillter

def hessian_ridge_response(image, sigma, darkline):
    img = img_as_float(image) 

    H_elems = hessian_matrix(img, sigma=sigma, order='xy') #give a hessian matrix with gausian filter (sigma defined in function below) 
    #good explanation here: 
    l1, l2 = hessian_matrix_eigvals(H_elems) #give a 2 eigenvalues

    abs1 = np.abs(l1)
    abs2 = np.abs(l2)
    swap = abs2 > abs1 #take the highest eugenvalue

    l1_swapped = l1.copy()
    l2_swapped = l2.copy()

    '''
    H, W = l1.shape

    for y in range(H):
        for x in range(W):
            if abs(l2[y, x]) > abs(l1[y, x]):
                l1_swapped[y, x] = l2[y, x]
                l2_swapped[y, x] = l1[y, x]
    '''
    #we make all eugenvalues l1 to be the highest to easier interpretation. Thus, l1 will represent the maximum curvature
    #ChatGPT rewrite faster way
    l1_swapped[swap], l2_swapped[swap] = l2[swap], l1[swap]

    #save only ridge-points 
    if darkline:
        response = np.maximum(0, -l1_swapped) #if we have darklines our valleys will have a positive value
    else:
        response = np.maximum(0, l1_swapped) #if we have bright lines, our valleys will have a negative value 

    resp = response - response.min() #make the lowest value 0
    maxv = resp.max() #
    if maxv > 0:
        resp = resp / maxv #make min-max normalization to [0,1]
    resp = (resp * 255).astype(np.float32) #transform back to 255 range; it needs to tresholding in ridge_mask_for_linewidth

    return resp

#function to process results of Ridge Detection by lineWidth

def ridge_mask_for_linewidth(
    image,
    lineWidth,
    im_mask,
    contrastLow,
    contrastHigh,
    darkline
):
    sigma = calcSigma(lineWidth)

    lower = calcLowerThresh(
        lineWidth, sigma, contrastLow, contrastHigh, darkline
    )
    upper = calcUpperThresh(
        lineWidth, sigma, contrastLow, contrastHigh, darkline
    )

    response = hessian_ridge_response(image, sigma, darkline=darkline)
    #exclude responce values that are in the image mask with background pixels on -np.inf 
    if im_mask is not None:
        response[im_mask] = -np.inf

    mask = (response >= lower) & (response <= upper)
    return mask, response, lower, upper

#old version, now works with skan

'''
# Curvature function (Rewrite with scan and calculate areas)
def curvature(coords, window=40):
    if len(coords) < window:
        return np.nan

    curves = []

    for i in range(0, len(coords) - window):
        seg = coords[i:i+window]
        (y0, x0) = seg[0]
        (y1, x1) = seg[-1]

        line_vec = np.array([x1 - x0, y1 - y0])
        line_len = np.linalg.norm(line_vec)
        if line_len == 0:
            continue

        line_unit = line_vec / line_len

        dists = []
        for (y, x) in seg:
            p = np.array([x - x0, y - y0])
            proj = np.dot(p, line_unit) * line_unit
            perp = p - proj
            dists.append(np.linalg.norm(perp))

        curves.append(np.mean(dists))

    return np.mean(curves) if curves else np.nan
'''

#extract skeleton coords by component
def get_component_coords(skel, comp_mask):
    #additional filtering, take only skeleton coords that are in the ridge-detection mask
    coords = np.argwhere(comp_mask & skel)

    # make everything ordered, not just a random order of coords
    coords = coords[np.lexsort((coords[:,1], coords[:,0]))] #like there https://stackoverflow.com/questions/60892374/sort-a-numpy-array-according-to-specific-column
    
    #return as list of tuples (y,x) 
    return [(int(y), int(x)) for (y,x) in coords]

# curvature extraction from skeleton (old version, now work with skan)

'''
def compute_curvature_from_skeleton(
    skeleton,
    windows=windows_curvature,
    min_pixels=minimumBranchLength
):
    labeled = label(skeleton)
    results = []

    for comp_id in range(1, labeled.max()+1):
        comp_mask = (labeled == comp_id)
        n_pix = comp_mask.sum()

        if n_pix < min_pixels:
            continue

        coords = get_component_coords(skeleton, comp_mask)

        row = {
            "component_id": comp_id,
            "n_pixels": n_pix,
        }

        # compute curvature for each window
        for w in windows:
            row[f"curvature_w{w}"] = curvature(coords, window=w)

        results.append(row)

    return pd.DataFrame(results)
'''

###### ---CURVATURE--- ######

import numpy as np
import pandas as pd
from skan import Skeleton, summarize


def curvature(coords, window=40):
    if len(coords) < window:
        return np.nan

    curves = []

    for i in range(0, len(coords) - window):
        seg = coords[i:i + window]
        (y0, x0) = seg[0]
        (y1, x1) = seg[-1]

        line_vec = np.array([x1 - x0, y1 - y0])
        line_len = np.linalg.norm(line_vec)
        if line_len == 0:
            continue

        line_unit = line_vec / line_len

        dists = []
        for (y, x) in seg:
            p = np.array([x - x0, y - y0])
            proj = np.dot(p, line_unit) * line_unit
            perp = p - proj
            dists.append(np.linalg.norm(perp))

        curves.append(np.mean(dists))

    return np.mean(curves) if curves else np.nan


def compute_curvature_from_skeleton(
    skeleton,
    windows,
    min_pixels
):
    sk = Skeleton(skeleton)
    summary = summarize(sk)

    results = []

    for branch_id, row in summary.iterrows():

        branch_length = row["branch-distance"]
        if branch_length < min_pixels:
            continue

        coords = sk.path_coordinates(branch_id)
        coords = [(int(y), int(x)) for y, x in coords]

        row_out = {
            "component_id": branch_id,
            "n_pixels": branch_length,
        }

        for w in windows:
            curv = curvature(coords, window=w)

            if not np.isnan(curv) and branch_length > 0:
                curv = curv / branch_length

            row_out[f"curvature_w{w}"] = curv

        results.append(row_out)

    return pd.DataFrame(results)

####### ---MAIN FUNCTION--- ######

def run_multiscale_ridge_detection(
    image,
    im_mask,
    minLineWidth,
    maxLineWidth,
    minimumBranchLength,
    do_skeleton,
    do_enhance_contrast,
    intensity_clip_percent,
    prune_short,
    contrastLow,
    contrastHigh,
    darkline
):
    #check that image is in uint8 format, if not - convert to uint8 for better workability of contrast enhancement and ridge detection.
    if image.dtype != np.uint8:
        img8 = (255 * (image - image.min()) / (image.max() - image.min())).astype(np.uint8)
    else:
        img8 = image

    #enhance contrast if needed
    if do_enhance_contrast:
        img = enhance_contrast(img8, intensity_clip_percent=intensity_clip_percent)
    else:
        img = img_as_float(img8)

    combined_mask = None

    # check if there is an image mask and convert it to boolean 
    if im_mask is not None:
      im_mask = im_mask.astype(bool)

    # frist LW
    lw = minLineWidth

    #start ridge detection for the first line width 
    #mask_lw is a boolean mask of detected ridge points after filtration and background pixels exclusion
    #responce is a responce array after ridgge detector after filtration and background pixels exclusion
    # lower and upper are thresholds for the ridge detector for this defined line widths 
    mask_lw, response, lower, upper = ridge_mask_for_linewidth(
    img, lw, im_mask,
    contrastLow, contrastHigh, darkline
    )
    #define new var to copy the mask for the first line width 
    combined_mask = mask_lw.copy()

    # other LW
    for lw in range(minLineWidth + 1, maxLineWidth + 1):
        mask_lw, response, lower, upper = ridge_mask_for_linewidth(
        img, lw, im_mask,
        contrastLow, contrastHigh, darkline
        )
        #Combining scales by logical OR to get the final mask of ridges of different widths
        combined_mask |= mask_lw

    
    if do_skeleton:
        skel = skeletonize(combined_mask)
        if prune_short:
                skel = remove_short_components(skel, minimumBranchLength)
        if np.count_nonzero(skel) == 0:
            skel = None
    else:
        skel = None


    return combined_mask, skel

def segmentation_features(
        im_list,
        windows_curvature, window_size, overlap,
        im_mask, intensity_thresh, intensity_clip_percent, eccentricity_thresh, minLineWidth, maxLineWidth,
        minimumBranchLength, do_skeleton, do_enhance_contrast, prune_short, contrastLow, contrastHigh, darkline
    ):

    #empty list to save records of features for each patch
    records = []

    for im_path in im_list:

        #define label for the image based on its filename
        label = os.path.splitext(os.path.basename(im_path))[0]

        #read image
        im = img_as_float(io.imread(im_path))
        if im.ndim == 3:
            im = im[0]

        #extract coords of patches
        x, y, u, v, im_theta, im_ecc = AFT.image_local_order(
            im, window_size, overlap,
            im_mask, intensity_thresh, eccentricity_thresh,
            plot_overlay=False, plot_angles=False, plot_eccentricity=False,
            save_figures=False, save_path=''
        )


        # check if there is an image mask
        if im_mask is None:
            im_mask = np.ones_like(im).astype('bool')
            #make all image valuable
        else:
            # make sure the input mask is a boolean
            im_mask = im_mask.astype('bool')


        mask, skel = run_multiscale_ridge_detection(
            im,
            im_mask,
            minLineWidth=minLineWidth,
            maxLineWidth=maxLineWidth,
            minimumBranchLength=minimumBranchLength,
            do_skeleton=do_skeleton,
            do_enhance_contrast=do_enhance_contrast,
            intensity_clip_percent=intensity_clip_percent,
            prune_short=prune_short,
            contrastLow=contrastLow,
            contrastHigh=contrastHigh,
            darkline=darkline
        )
        if skel is None:
            continue
        # extract X,Y coords
        x_flat = np.ravel(x).astype(int) #extract x patch coords and make it 1D array
        y_flat = np.ravel(y).astype(int) #extract y patch coords and make it 1D array

        half = window_size // 2 #half of the patch size 
        H, W = im.shape #extract height and width of the image

        #take all overlap coordinates in x_flat and y_flat
        for xx, yy in zip(x_flat, y_flat):
            xx_i = int(xx) #take x coord
            yy_i = int(yy) #take y coord

            x1 = xx_i - half #left border
            x2 = xx_i + half #right border
            y1 = yy_i - half #bottom border
            y2 = yy_i + half #top border
            
            if skel is None:
                continue 
            
            if x1 < 0 or y1 < 0 or x2 > W or y2 > H:
              continue #skip patches that are out of image borders
            else:
              patch = skel[y1:y2, x1:x2] #extract skeleton from patch 
              if patch is None or np.count_nonzero(patch) == 0:
                continue
              im_patch = im[y1:y2, x1:x2] #extract image patch for not skeleton based features
              mask_patch = mask[y1:y2, x1:x2] #extract mask patch for Ridge-Detector based features
              
              #Intensity
              mean_int = float(np.mean(im_patch))
              
              #HDM
              HDM_value = np.count_nonzero(mask_patch) / mask_patch.size

              #skeleton graph
              from skan import Skeleton, summarize
              if np.count_nonzero(patch) == 0:
                continue
              
              if patch is None or np.count_nonzero(patch) == 0:
                continue
              patch_bool = patch.astype(bool)
              if np.count_nonzero(patch_bool) == 0:
                continue

              try:
                sk = Skeleton(patch_bool, spacing=1)
                branch_data = summarize(sk, separator='_')
              except ValueError:
                continue

              patch_bool = patch.astype(bool)
              #if np.count_nonzero(patch_bool) == 0:
                #continue

              #branch_data = summarize(Skeleton(patch_bool, spacing=1), separator='_')
              branch_data = branch_data.drop_duplicates()

              # endpoints
              branch_types = branch_data['branch_type'].value_counts()

              m = 0
              m += branch_types.get(0, 0) * 2  # separate branch (two endpoints)
              m += branch_types.get(1, 0) * 1  # endpoints

              #branch points
              sk = Skeleton(patch, spacing=1)
              G = skeleton_to_nx(sk)
              branch_point = []
              for i in G.degree:
                branch_point.append(i[1] == 3)
              n_branchpoints = branch_point.count(True)
              #print(branch_points)

              # normalization
              if branch_data['branch_distance'].sum() > 0:
                  norm_end = m / branch_data['branch_distance'].sum()
                  norm_branch = n_branchpoints / branch_data['branch_distance'].sum()
              else:
                  norm_end = np.nan
                  norm_branch = np.nan

              #ffd
              H_patch, W_patch = patch.shape
              ys, xs = np.where(patch > 0)
              x_norm = xs / (W_patch - 1)
              y_norm = ys / (H_patch - 1)
              points = np.column_stack([x_norm, y_norm])

              scales = np.logspace(np.log10(0.02),np.log10(0.25), 5)
              result = box_counting(points, scales, method="oversample")

              #print("Fractal Dimension:", result["fd"])

              #lacunarity (use the formula drectly from TWOMBLI)
              lac_value = abs((((im_patch.std())**2)/((im_patch.mean())**2)) - 1)

              #curvature
              curv_values = []

              for w in windows_curvature:
                if w <= window_size:
                    df_curv = compute_curvature_from_skeleton(
                        patch,
                        windows=[w],
                        min_pixels=minimumBranchLength
                    )
                    col = f"curvature_w{w}"
                    if col in df_curv.columns:
                        curv_values.append(df_curv[col].mean())
              curvature_mean = np.mean(curv_values) if curv_values else np.nan

              #record
              records.append({
                  "image_name": label,
                  "ECM_x": xx_i,
                  "ECM_y": yy_i,
                  "intensity": mean_int,
                  "HDM": HDM_value,
                  "endpoints": m,
                  "norm_endpoints": norm_end,
                  "norm_branch": norm_branch,
                  "curvature_mean": curvature_mean,
                  "branch_points": n_branchpoints,
                  "FFD": result["fd"],
                  "lacunarity": lac_value,
              })

    return pd.DataFrame(records)
