import os
import numpy as np
from glob import glob
from tqdm import tqdm
from skimage.morphology import convex_hull_image, disk, closing
from skimage.measure import label, regionprops, regionprops_table
import math
from skimage.draw import line_aa
import pandas as pd
import ast

from utils import util_seg

def gen_df_measurements(filenames, filename_csv, model, view):
    """
    Generate dataframe of measurements for each echo chamber.
    
    Input: 
        filenames: list of paths to dicom files
        filename_csv: path to save csv of measurements
        view = 'a2c' or 'a4c'
    """

    if view == 'A2C':
        list_chambers = ['LA', 'LV']
    else:
        list_chambers = ['LA', 'LV', 'RA', 'RV']

    df_measures = pd.DataFrame([])

    with tqdm(total=len(filenames)) as pbar:
        for filename in filenames:
                
            val = os.path.basename(filename)
            img_fov = np.load(filename)

            if view == 'SAXMID':
                # Convert to grayscale
                img_fov = img_fov[..., 0:1]
            
            pred = model.predict(img_fov[np.newaxis])

            try:
                pred = util_seg.refine_chambers(pred[0], view)
            except:
                print('ERROR: fail to predict this image:', filename)
                continue
                
            if view == 'SAXMID':
                pred_ = pred.copy()
                pred_[...,1] = pred[...,0]
                pred_[...,0] = pred[...,1]
                pred = pred_.copy()
                list_chambers = ['inner_circle', 'outer_circle']
                
            ### Compute measurements for each chamber
            for c, chamber in enumerate(list_chambers):
                try:
                    data = DataframeOfMeasures(pred[..., c+1])
                except:
                    data = pd.DataFrame([])
                    continue

                data['label'] = chamber
                data['ImgName'] = val
                data['anonid'] = val.split('_')[0]
                data['ID_clip'] = val.split('_')[0] + '_'  + val.split('_')[1]
                data['frame'] = val.split('_')[2].split('.')[0]

                df_measures = df_measures.append(data)

                with open(filename_csv, 'a') as f:
                    pd.DataFrame(data).to_csv(f, header=f.tell()==0, index=False)

            pbar.update(1)

    return df_measures

    
def get_props(mask):
    """
    measure properties of labelled image regions. 
    Input: 
        mask: Labeled input image.
    Output: 
        Dataframe of 'label','bbox','area', 'filled_area', 'convex_area', 'centroid','eccentricity',
        'orientation','major_axis_length','minor_axis_length', 'solidity (Ratio of pixels in the region to pixels of the convex hull image)' 
            
    Obs: 'centroid-0' = row, 'centroid-1' = column

    """
    props = regionprops_table(mask, properties=('bbox','area', 'filled_area', 'convex_area', 'centroid','eccentricity',
                                                'orientation','major_axis_length','minor_axis_length', 'solidity'))
    table_props = pd.DataFrame(props)
    
    pixel_space = (400/mask.shape[0] * 0.05)
    pixel_space_squared = pixel_space ** 2 

    table_props[['area', 'filled_area', 'convex_area']] = table_props[['area', 'filled_area', 'convex_area']] * pixel_space_squared


    return table_props 


def DataframeOfMeasures(mask):
    """
    Generate dataframe of measures from echo predictions.
    Input: 
        mask: image prediction.
    Output: Dataframe of measurements including chamber's area, height, disks, eccentricity, etc.
    """   
    if len(mask.shape) != 2:
        print(f'ERROR: wrong mask shape. Expected 2 got {len(mask.shape)}')
    pixel_space = (400/mask.shape[0] * 0.05)
    data = get_props(mask)
    data[['L', 'L_model', 'img_h', 'x_top', 'y_top']] = pd.DataFrame(get_chamber_length(mask, pixel_space)).T
    data[['D2', 'L_disk', 'img_disk',  'x_left', 'y_left', 'x_right', 'y_right']] = pd.DataFrame([get_img_disks_values(mask,pixel_space,21)])
    data.drop(['img_h','img_disk'], axis=1, inplace=True)
    return data
    

def get_chamber_length(image, pixel_space=False):
    """
    Extract the linear measurements of the chamber.
    Input: A 2D array image
    return: chamber_length, 2D binary image containing chamber length line
    """
    
    # Use regionprops to extract certain properties of the structure, such as 
    #orientation, centroid and the major and minor axes of an ellipse.
 
    label_img = label(image)
    props = regionprops(label_img)
    
    # Find the centroid coordinate tuple (row,col).
    y0, x0 = props[0].centroid
    orientation = props[0].orientation
    
    if orientation > 0.7:
        orientation = 0.45
    if orientation < (-0.7):
        orientation = - 0.45
       

    # Coordinates of chamber length. Find the angle between the 0th axis (rows) and the major axis of the ellipse, 
    # ranging from -pi/2 to pi/2 counter-clockwise.
    x_top = (x0 - math.sin(orientation) * 0.6 * props[0].major_axis_length).astype(int)
    y_top = (y0 - math.cos(orientation) * 0.6 * props[0].major_axis_length).astype(int)
    x_bot = (x0 + math.sin(orientation) * 0.6 * props[0].major_axis_length).astype(int)
    y_bot = (y0 + math.cos(orientation) * 0.6 * props[0].major_axis_length).astype(int)
    
        
    ## Draw a line connecting the two end points. 
    img_chamber_length = np.zeros_like(image, dtype=np.uint8)

    rr, cc, val = line_aa(y_bot, x_bot, y_top, x_top)
    
    # mask check here, because line_aa can generate coordinates outside of the image plane

    rr_mask = np.logical_and(0 <= rr, rr < image.shape[0])
    cc_mask = np.logical_and(0 <= cc, cc < image.shape[1])
    mask = np.logical_and(rr_mask, cc_mask)

    if np.any(mask):
        rr = rr[mask]
        cc = cc[mask]

    
    img_chamber_length[rr, cc] = 1
    
    # Take only points inside the structure.
    img_mask = image.copy()
    end_points = np.where((img_chamber_length * img_mask) > 0)
    
    # Compute chamber length.
    
    chamber_length = math.hypot(end_points[1][-1] - end_points[1][0], end_points[0][-1] - end_points[0][0])
    
    y_top_masked, x_top_masked = end_points[1][0], end_points[0][0]
    
    if pixel_space:
        chamber_length_cm = chamber_length * pixel_space
    else:
        chamber_length_cm = None
            
    return chamber_length, chamber_length_cm, img_chamber_length * img_mask, x_top_masked, y_top_masked


def get_img_disks_values(image, pixel_spacing, n = 21):
    """
    Calculate the values of image disks.
    Parameters:
    image (ndarray): The input image.
    pixel_spacing (float): The pixel spacing of the image.
    n (int): The number of disks.
    Returns:
    tuple: A tuple containing the following values:
        - v_d1 (list): The list of disk diameters.
        - h_cm (float): The height of each disk in centimeters.
        - img (ndarray): The modified image with disks and other structures.
        - v_x_left (list): The list of x-coordinates of the left endpoints of the disks.
        - v_y_left (list): The list of y-coordinates of the left endpoints of the disks.
        - v_x_right (list): The list of x-coordinates of the right endpoints of the disks.
        - v_y_right (list): The list of y-coordinates of the right endpoints of the disks.
    """

    
    # Use regionprops to extract certain properties of the structure, such as 
    #orientation, centroid and the major and minor axes of an ellipse.

    label_img = label(image)
    props = regionprops(label_img)
    
    # Parameters to be used in BI-PLANE METHOD OF DISK SUMMATION. 
    L, L_cm, img_h, x_top, y_top = get_chamber_length(image, pixel_spacing)  #  total length of the left ventricle 
    
    # Find the centroid coordinate tuple (row,col).
    x0, y0 = y_top, x_top
    orientation = props[0].orientation
    
    if orientation > 0.7:
        orientation = 0.45
    if orientation < (-0.7):
        orientation = - 0.45
        
    # Coordinates of the LV diameter. Find the angle between the 0th axis (rows) and the major axis of the ellipse, 
    # ranging from -pi/2 to pi/2 counter-clockwise.
    if orientation < 0:
        x0_left = (x0 - math.cos(orientation) *  image.shape[1]/2).astype(int)
        y0_left = (y0 + math.sin(orientation) *  image.shape[1]/2).astype(int)
        x0_right = (x0 + math.cos(orientation) *  image.shape[1]/2).astype(int)
        y0_right = (y0 - math.sin(orientation) *  image.shape[1]/2).astype(int)
    else:
        x0_left = (x0 + math.cos(orientation) *  image.shape[1]/2).astype(int)
        y0_left = (y0 - math.sin(orientation) *  image.shape[1]/2).astype(int)
        x0_right = (x0 - math.cos(orientation) *  image.shape[1]/2).astype(int)
        y0_right = (y0 + math.sin(orientation) *  image.shape[1]/2).astype(int)
    
    
    # number of disks 
    h = L  / n  # height of each disk
    h_cm = L_cm/n
    img_disk2 = np.zeros_like(image, dtype=np.uint8)
    v_d1 = []
    v_x_left = []
    v_y_left = []
    v_x_right = []
    v_y_right = []
    
    for i in range(1,n+1):
        try: 
            x1_right = int((x0_right ) +  h  )
            y1_right = int((y0_right ) + i * h / math.cos(orientation))
            x1_left = int((x0_left ) +  h )
            y1_left = int((y0_left ) + i * h / math.cos(orientation) )
        
            rr, cc, val = line_aa(y1_right,x1_right,y1_left,x1_left) 

            # mask check here, because line_aa can generate coordinates outside of the image plane

            rr_mask = np.logical_and(0 <= rr, rr < image.shape[0])
            cc_mask = np.logical_and(0 <= cc, cc < image.shape[1])
            mask = np.logical_and(rr_mask, cc_mask)

            if np.any(mask):
                rr = rr[mask]
                cc = cc[mask]


            img_disk = np.zeros_like(image, dtype=np.uint8)

            img_disk[rr, cc] = 1
            img_disk2[rr, cc] = 1

            end_points = np.where((img_disk * image ) > 0)

            x_left, y_left = end_points[1].min().astype(int), (end_points[0][end_points[1] == end_points[1].min()][0]).astype(int)
            x_right,y_right = end_points[1].max().astype(int), (end_points[0][end_points[1] == end_points[1].max()][0]).astype(int)

            D1 = math.hypot(x_left-x_right,y_left-y_right) * pixel_spacing

            v_d1.append(D1)
            v_x_left.append(y_left)
            v_y_left.append(x_left)
            v_x_right.append(y_right)
            v_y_right.append(x_right)
            
        except:
            continue

    return v_d1, h_cm, (img_disk2 *  image) + img_h + image, v_x_left, v_y_left, v_x_right, v_y_right


def get_volumetrics(df, metric):
    """
    Calculate volumetric measurements based on the given metric.
    Parameters:
    - df (pandas.DataFrame): The input dataframe containing the necessary columns.
    - metric (str): The metric to calculate. Available options: 'LV_volumetrics', 'LAVI', 'RAVI', 'LV_mass'.
    Returns:
    - df (pandas.DataFrame): The input dataframe with the calculated volumetric measurements added as new columns.
    """
    

    if metric == 'LV_volumetrics':

        df['LVEDV_model'] = 0
        df['LVESV_model'] = 0
        df['LVEDVI_model'] = 0
        df['LVESVI_model'] = 0

        for i in tqdm(range(df.shape[0])):

            # Get the left ventricle length and disks for diastole and systole

            ## Diastole
            D2c_d_LV = np.array(ast.literal_eval(df.D2_dia_a2c.iloc[i]))
            L_2c_d_LV = df.L_model_dia_a2c.iloc[i] / 21
            D4c_d_LV = np.array(ast.literal_eval(df.D2_dia_a4c.iloc[i]))
            L_4c_d_LV = df.L_model_dia_a4c.iloc[i] / 21

            ## Systole
            D2c_s_LV = np.array(ast.literal_eval(df.D2_sys_a2c.iloc[i]))
            L_2c_s_LV = df.L_model_sys_a2c.iloc[i] / 21
            D4c_s_LV = np.array(ast.literal_eval(df.D2_sys_a4c.iloc[i]))
            L_4c_s_LV = df.L_model_sys_a4c.iloc[i] / 21


            # Calculate the minimum length for diastole and systole
            r_d_LV = min(len(D2c_d_LV), len(D4c_d_LV))
            r_s_LV = min(len(D2c_s_LV), len(D4c_s_LV))

            # Calculate the volume of the left ventricle in diastole and systole
            vol_d_LV = []
            for k in range(r_d_LV):        
                vol_d_LV.append(D2c_d_LV[k] * D4c_d_LV[k] * (max(L_2c_d_LV, L_4c_d_LV)))
            vol_d_LV_sum = ((np.pi * np.sum(vol_d_LV)) / 4)

            vol_s_LV = []
            for k in range(r_s_LV):        
                vol_s_LV.append(D2c_s_LV[k] * D4c_s_LV[k] * (max(L_2c_s_LV, L_4c_s_LV)))
            vol_s_LV_sum = ((np.pi * np.sum(vol_s_LV)) / 4)

            BSA = df.BSA.iloc[i]
            df['LVEDV_model'].iloc[i] = vol_d_LV_sum 
            df['LVESV_model'].iloc[i] = vol_s_LV_sum 
            df['LVEDVI_model'].iloc[i] = vol_d_LV_sum / BSA
            df['LVESVI_model'].iloc[i] = vol_s_LV_sum / BSA

        return df

    elif metric == 'LAVI':

        df['LAVI_model'] = 0

        for i in tqdm(range(df.shape[0])):

            # Get the left atrium length and disks
            D2c_LA = np.array(ast.literal_eval(df.D2_a2c.iloc[i]))
            L_2c_LA = df.L_model_a2c.iloc[i] / 21
            D4c_LA = np.array(ast.literal_eval(df.D2_a4c.iloc[i]))
            L_4c_LA = df.L_model_a4c.iloc[i] / 21

            # Calculate the minimum length
            r_LA = min(len(D2c_LA), len(D4c_LA))

            # Calculate the volume of the left atrium
            vol_LA = []
            for k in range(r_LA):        
                vol_LA.append(D2c_LA[k] * D4c_LA[k] * (max(L_2c_LA, L_4c_LA)))
            vol_sum = ((np.pi * np.sum(vol_LA)) / 4)

            BSA = df.BSA.iloc[i]
            df['LAVI_model'].iloc[i] = vol_sum / BSA
            
        return df
    
    elif metric == 'RAVI':

        df['RAVI_model'] = 0

        for i in tqdm(range(df.shape[0])):

            # Get the right atrium length and disks
            D4c_RA = np.array(ast.literal_eval(df.D2.iloc[i]))
            L_4c_RA = df.L_model.iloc[i] / 21

            # Calculate the minimum length
            r_RA = len(D4c_RA)

            # Calculate the volume of the right atrium
            vol_RA = []
            for k in range(r_RA):        
                vol_RA.append(D4c_RA[k] * D4c_RA[k] * L_4c_RA)
            vol_sum = ((np.pi * np.sum(vol_RA)) / 4)

            BSA = df.BSA.iloc[i]
            df['RAVI_model'].iloc[i] = vol_sum / BSA
            
        return df
    
    elif metric == 'LV_mass':

        df['LVMI_model'] = 0

        for i in tqdm(range(df.shape[0])):

            # Get the A 1 area, A 2 area, and the length of the left ventricle
            h_model = max(df.L_model_a2c[i], df.L_model_a4c[i])
            lvmass = get_lv_mass(df.area_A1.iloc[i], df.area_A2.iloc[i], h_model)
            bsa = df.BSA.iloc[i]
            df['LVMI_model'].iloc[i] = lvmass / bsa
            
        return df


def get_ef(EDV, ESV):
    """
    Left ventricular ejection fraction (LVEF) 

    SV (stroke volume): Stroke volume (SV) is calculated as the difference between end-diastolic volume (EDV) and end-systolic volume (ESV) 
    LVEF: [SV/EDV] x 100
    """
    return ((EDV - ESV)/EDV) * 100


def get_lv_mass(A1, A2, h):
    """
    Calculate the left ventricular (LV) mass.
    Parameters:
    A1 (float): Area of the first cross-section of the LV.
    A2 (float): Area of the second cross-section of the LV.
    h (float): Height of the LV.
    Returns:
    float: The calculated LV mass.
    """
    
    b = np.sqrt(A2/np.pi)
    t = np.sqrt(A1/np.pi) - b
    LV_mass = 1.05 * ((5/6 * (A1 * (h + t))) - (5/6 * (A2 * (h))))
    return LV_mass


def gen_qc_measures(base_from, view='A4C'):
    """
    Generate quality control measures for cardiac segmentation.
    Parameters:
    - base_from (str): The base directory path where the .npy files are located.
    - view (str): The cardiac view to generate measures for. Default is 'A4C'.
    Returns:
    - df_measures (pandas.DataFrame): The dataframe containing the quality control measures.
    Raises:
    - None
    Example usage:
    df = gen_qc_measures('/home/ec2-user/ssl-seg-model-cleaned/utils/util_measurement.py', view='A2C')
    """

    df_measures = pd.DataFrame([])

    with tqdm(total=len(glob(os.path.join(base_from, '*.npy'))) ) as pbar:

        for filename in glob(os.path.join(base_from, '*.npy'),recursive = False):

            pred = np.load(filename)[np.newaxis]
            pixel_space = 400/pred.shape[1] * 0.05
            th = 0.5

            try:
                # Apply morphological closing to the largest connected component of the segmented image
                img_LA = closing((util_seg._get_largest_component((pred[0, ..., 1] > th))).astype(int), disk(2))
                
                # Compute the convex hull of the closed image
                img_LA_hull = convex_hull_image(img_LA)
                
                # Calculate the area of the left atrium segmentation
                LA_A_seg = img_LA.sum() * (pixel_space) ** 2
                
                # Calculate the area of the convex hull of the left atrium segmentation
                LA_A_seg_hull = img_LA_hull.sum() * (pixel_space) ** 2
                
                # Calculate the difference in area between the convex hull and the original segmentation
                LA_diff_hull = LA_A_seg_hull - LA_A_seg
                
                # Get the length of the left atrium chamber
                _, LA_L_seg, _, _, _ = get_chamber_length(img_LA, pixel_space)
                
                # Calculate the eccentricity of the left atrium segmentation
                ecc_LA = regionprops(img_LA.astype(int))[0]['eccentricity']

            except:
                # If an error occurs during LA measurements, set the values to 0
                LA_A_seg, LA_A_seg_hull, LA_L_seg, ecc_LA , LA_diff_hull = 0, 0, 0, 0, 0

            try:
                # Segment the left ventricle
                img_LV = closing((util_seg._get_largest_component((pred[0,...,2] > th))).astype(int),disk(2))
                img_LV_hull = convex_hull_image(img_LV)
                
                # Calculate the area of the left ventricle segmentation
                LV_A_seg = img_LV.sum() * (pixel_space) ** 2 
                
                # Calculate the area of the convex hull of the left ventricle segmentation
                LV_A_seg_hull = img_LV_hull.sum() * (pixel_space ) ** 2
                
                # Calculate the difference in area between the convex hull and the original segmentation
                LV_diff_hull = LV_A_seg_hull - LV_A_seg
                
                # Get the length of the left ventricle chamber
                _, LV_L_seg, _, _, _ = get_chamber_length(img_LV, pixel_space)
                
                # Calculate the eccentricity of the left ventricle segmentation
                ecc_LV = regionprops(img_LV.astype(int))[0]['eccentricity']

            except:
                # If an error occurs during LV measurements, set the values to 0
                LV_A_seg, LV_A_seg_hull, LV_L_seg, ecc_LV , LV_diff_hull = 0, 0, 0, 0, 0

            if view == 'A4C':

                try:
                    # Apply morphological closing to the largest connected component of the segmented region
                    img_RA = closing((util_seg._get_largest_component((pred[0,...,3] > th))).astype(int), disk(2))

                    # Compute the convex hull of the closed image
                    img_RA_hull = convex_hull_image(img_RA)

                    # Calculate the area of the segmented region in square pixels
                    RA_A_seg = img_RA.sum() * (pixel_space) ** 2 

                    # Calculate the area of the convex hull of the segmented region in square pixels
                    RA_A_seg_hull = img_RA_hull.sum() * (pixel_space) ** 2 

                    # Calculate the difference in area between the convex hull and the segmented region
                    RA_diff_hull = RA_A_seg_hull - RA_A_seg

                    # Get the chamber length and other properties of the segmented region
                    _, RA_L_seg, _, x_top, y_top = get_chamber_length(img_RA, pixel_space)

                    # Calculate the eccentricity of the segmented region
                    ecc_RA = regionprops(img_RA.astype(int))[0]['eccentricity']

                except: 
                    # If an error occurs during RA measurements, set the values to 0
                    RA_A_seg, RA_A_seg_hull, RA_L_seg, ecc_RA, RA_diff_hull = 0, 0, 0, 0, 0

                try:
                    # Apply morphological closing to the largest connected component of the RV prediction
                    img_RV = closing((util_seg._get_largest_component((pred[0,...,4] > th))).astype(int), disk(2))

                    # Compute the convex hull of the RV image
                    img_RV_hull = convex_hull_image(img_RV)

                    # Calculate the area of the segmented RV in square pixels
                    RV_A_seg = img_RV.sum() * (pixel_space) ** 2

                    # Calculate the area of the convex hull of the segmented RV in square pixels
                    RV_A_seg_hull = img_RV_hull.sum() * (pixel_space) ** 2

                    # Compute the difference in area between the convex hull and the segmented RV
                    RV_diff_hull = RV_A_seg_hull - RV_A_seg

                    # Get the length of the RV chamber and other measurements
                    _, RV_L_seg, _, _, _ = get_chamber_length(img_RV, pixel_space)

                    # Calculate the eccentricity of the RV region
                    ecc_RV = regionprops(img_RV.astype(int))[0]['eccentricity']

                except: 
                    # If an error occurs during RV measurements, set the values to 0
                    RV_A_seg, RV_A_seg_hull, RV_L_seg, ecc_RV , RV_diff_hull = 0, 0, 0, 0, 0

            if view == 'A2C':
                data = pd.DataFrame({
                    'Image': os.path.basename(filename),
                    'LV_A_seg':LV_A_seg, 'LA_A_seg':LA_A_seg,
                    'LV_L_seg':LV_L_seg, 'LA_L_seg':LA_L_seg,
                    'ecc_LV':ecc_LV, 'ecc_LA':ecc_LA,
                    'LV_diff_hull':LV_diff_hull, 'LA_diff_hull':LA_diff_hull
                    }, index=[0])
            else:
                data = pd.DataFrame({
                    'Image': os.path.basename(filename),
                    'LV_A_seg':LV_A_seg, 'LA_A_seg':LA_A_seg, 'RV_A_seg':RV_A_seg, 'RA_A_seg':RA_A_seg,
                    'LV_L_seg':LV_L_seg, 'LA_L_seg':LA_L_seg, 'RV_L_seg': RV_L_seg, 'RA_L_seg':RA_L_seg, 
                    'ecc_LV':ecc_LV, 'ecc_LA':ecc_LA, 'ecc_RV':ecc_RV, 'ecc_RA':ecc_RA,
                    'LV_diff_hull':LV_diff_hull, 'LA_diff_hull':LA_diff_hull, 'RV_diff_hull':RV_diff_hull, 'RA_diff_hull':RA_diff_hull
                    }, index=[0])

            df_measures = df_measures.append(data)

            pbar.update(1)


    return df_measures
