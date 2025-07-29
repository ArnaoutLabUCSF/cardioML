import numpy as np
import pandas as pd
import cv2
from scipy import ndimage as ndi
from skimage.segmentation import watershed, flood_fill
from skimage.measure import label, regionprops_table
from skimage.draw import polygon, line_aa
from skimage.restoration import denoise_wavelet, denoise_bilateral
from skimage.feature import corner_peaks, corner_harris, peak_local_max
import alphashape
from PIL import Image, ImageDraw
from skimage import morphology as morph
from skimage import img_as_float
import os

from utils import util_measurement


def get_img_pred(model, img_res, view):
    """
    Predicts the labels for an input image using a given model.
    Parameters:
    - model: The segmentation model used for prediction.
    - img_res: The input image to be predicted.
    - view: The view type for prediction.
    Returns:
    - label: The predicted labels for the input image.
    """

    pred = model.predict(img_res[np.newaxis,...])
    label = np.zeros_like(pred)
    if view == 'a2c':
        for k in range(3):
            label[0,...,k] = (pred[0,...,k]>0.5).astype(int)
    else:
        for k in range(5):
            try:
                label[0,...,k] = morph.dilation(_get_largest_component((pred[0,...,k]>0.5).astype(int)))
            except:
                label[0,...,k] = (pred[0,...,k]>0.5).astype(int)
    return label


def PartialconvexHull(label, side = 'left'):
    """
    construct a convex hull from the upper triangle of the chamber. 
    Input:
        img_h: 2D image of chamber height. Chamber height will be used to generate a triangle mask to create the partial convex hull.
        label: chamber's mask.
    Output:
        Partial convex hull of the label.
    """

    pixel_space = 400/label.shape[0] * 0.05

    _, _, img_h, _, _ = util_measurement.get_chamber_length(label,pixel_space)
    
    rr, cc, _ = line_aa(img_h.shape[0]-1, np.where(img_h == 1)[1][-1], 0, np.where(img_h == 1)[1][0])
    
    mask = np.zeros_like(img_h, dtype=np.uint8)
    mask[rr, cc] = 1

    mask = flood_fill(mask, (round(img_h.shape[0]/2), img_h.shape[0]-1), 1)
    
    if side == 'left':
        new_seg = (1-mask)*label + mask*morph.convex_hull_image(label)
    else:
        new_seg = (mask)*label + (1-mask)*morph.convex_hull_image(label)
    return new_seg


def refine_chambers(pred, view='A4C', PartialHull=False, AlphaShape=False):
    """
    Refines the chambers in the predicted image.
    Parameters:
    - pred (ndarray): The predicted image.
    - view (str): The view of the image. Default is 'A4C'.
    Returns:
    - ndarray: The refined chambers image.
    """
    
    if len(pred.shape) == 4:
        pred = pred[0]
    ch = pred.shape[-1]
    
    img_zero = np.zeros_like(pred[...,0], dtype=np.bool)

    for i in range(ch):

        try:
            pred[...,i] = morph.closing((_get_largest_component((pred[...,i] > 0.5))).astype(int), morph.disk(2)).astype(int)
            pred[...,i] = ndi.binary_fill_holes(pred[...,i].astype(int)).astype('uint8')
            if PartialHull == True:
                pred[...,i] = PartialconvexHull(pred[...,i], side='left')

            elif AlphaShape == True:
                pred[...,i] = get_alphashape(pred[...,i], alpha_=3)
        except:
            pred[...,i] = img_zero
            pass

    if view != 'SAXMID':
        if ch == 5:
            pred[...,0] = pred[...,1] + pred[...,2] + pred[...,3] + pred[...,4]
        else:
            pred[...,0] = pred[...,1] + pred[...,2]
        pred[...,0][pred[...,0] > 1] = 0

        for i in range(1,ch):
            pred[...,i] = pred[...,i] * (pred[...,0])

    return pred.astype('uint8')


def random_crop(img, mask, width, height):
    """
    Randomly crops an image and its corresponding mask to the specified width and height.
    Args:
        img (numpy.ndarray): The input image.
        mask (numpy.ndarray): The corresponding mask.
        width (int): The width of the cropped image.
        height (int): The height of the cropped image.
    Returns:
        tuple: A tuple containing the cropped image and its corresponding mask.
    """

    assert img.shape[0] >= height
    assert img.shape[1] >= width
    assert img.shape[0] == mask.shape[0]
    assert img.shape[1] == mask.shape[1]
    x = np.random.randint(0, img.shape[1] - width)
    y = np.random.randint(0, img.shape[0] - height)
    img = img[y:y+height, x:x+width]
    mask = mask[y:y+height, x:x+width]
    return img, mask

def cat_to_color(cat_array, color_list):
    """
    Changes categorical predictions to color segmentation  
    input: array of predicted labels, dim: (nb images, nrow, ncol, nb seg),
        list of color tuples for each segment, i.e. (125, 135, 255) lightred
    returns: array of COLOR predictions, dim: (nb of images, nrow, ncol, nb channels = 3)
    """
    nb_img, h, w, nb_seg = cat_array.shape
    color_array = np.zeros((nb_img*h*w, 3))
    index_array = cat_array.reshape(-1, nb_seg)

    for i, color in enumerate(color_list):
        color_index = index_array[...,i].astype(bool)
        color_array[color_index] = list(color)

    color_array = color_array.reshape(nb_img, h, w, 3)
    return color_array


def get_label_ws(image, min_distance=30, c=0, l=150, r=200):
    """
    Apply watershed segmentation to an input image.
    Parameters:
    - image: ndarray
        The input image to be segmented.
    - min_distance: int, optional
        The minimum distance between local maxima in the distance transform. Default is 30.
    - c: int, optional
        The value used to determine the separation of structures in the image. Default is 0.
    - l: int, optional
        The lower threshold value for removing small objects. Default is 150.
    - r: int, optional
        The upper threshold value for removing small objects. Default is 200.
    Returns:
    - labels_ws: ndarray
        The watershed labels of the segmented image.
    - img_seg: ndarray
        The segmented image after applying morphological operations.
    - img_props: ndarray
        The image properties after applying morphological operations.
    - img_prop_cat: ndarray
        The image properties after categorizing the segmented regions.
    - df: DataFrame
        The dataframe containing the properties of the watershed segments.
    """
    

    ## Remove small objects and small holes
    image = (image-image.min())/(image.max()-image.min())
    mask = morph.remove_small_holes(morph.remove_small_objects(image < 0.1, 500), 1000)
    mask = morph.opening(mask, morph.disk(2))
    
    # Separate the two structures in the image
    # Generate the markers as local maxima of the distance to the background    
    distance = ndi.distance_transform_edt(mask)
    local_maxi = peak_local_max(distance, indices=False , min_distance = min_distance )# footprint=np.ones((3, 3)), labels=image)
    markers = label(local_maxi)

    ## Generate the watershed segments
    labels_ws = watershed(-distance, markers, mask=mask)
    
    ## Create a dataframe with ws segments information
    props = regionprops_table(labels_ws, properties=('label','bbox','area','perimeter','local_centroid','centroid',
                                                 'orientation',
                                                 'major_axis_length',
                                                 'minor_axis_length'))
    df = pd.DataFrame(props)
    df.sort_values(by='area',ascending=False,inplace=True)
    
    # Remove segments that do not belong to the chamber using centroid info
    df = df[df['perimeter'] < 550]
    df = df[df['centroid-1'] > 75]
    df = df[df['centroid-1'] < 200]
    
    #display(df)
    if  len(df['bbox-1']) > 0:
        bb1 = df['bbox-1'].values[0]
        bb3 = df['bbox-3'].values[0]
        df = df[df['centroid-1'] < bb3+20]
        df = df[df['centroid-1'] > bb1-20]
        
        img_props = sum([labels_ws==df.label.values[i] for i in range(len(df.label.values))])
        img_binary = img_props
        img_props = img_props * labels_ws
        
        img_seg = morph.closing(img_props, morph.disk(4))
        img_seg = morph.remove_small_holes(img_seg, 2000)
    else:
        img_seg = []
        img_binary = []
        img_props = np.zeros((labels_ws.shape[0], labels_ws.shape[1], 3))
        print('bb1 empty:',len(df['bbox-1']))
    
    img_prop_cat = np.zeros((labels_ws.shape[0], labels_ws.shape[1], 3))

    for i, cent in enumerate(df['centroid-0'].values):

        if (cent < (labels_ws.shape[0]/2 + c)): 
            img_props[img_props == df.label.values[i]] = 200
            img_prop_cat[...,2] = (img_props == 200).astype('int')
            img_prop_cat[...,2] = _clean_segmask(img_prop_cat[...,2].astype(int),1)

        else:
            img_props[img_props == df.label.values[i]] = 180
            img_prop_cat[...,1] = (img_props == 180).astype('int')
            img_prop_cat[...,1] = _clean_segmask(img_prop_cat[...,1].astype(int),1)
        img_prop_cat[...,0] = img_prop_cat[...,1] + img_prop_cat[...,2]
    
    return labels_ws, img_seg, img_props, img_prop_cat, df


def unsharp_mask_filter(image, radius, amount, vrange, sigma_color, sigma_spatial,filter_,):
    """
    Single channel implementation of the unsharp masking filter.
    """

    if filter_ == 'gaussian':
        blurred = ndi.gaussian_filter(image, sigma=radius, mode='reflect')
    elif filter_ == 'wavelet':
        blurred = denoise_wavelet(image, sigma=radius, rescale_sigma=True) # multichannel=False, 
    elif filter_ == 'bilateral':
        blurred = denoise_bilateral(image, sigma_color=sigma_color, sigma_spatial= sigma_spatial) #, multichannel=False)
        
    result = image + (image - blurred) * amount
    
    result_clip = np.clip(result, vrange[0], vrange[1])
    
    print(image.max(), result.max(), result_clip.max())
    if vrange is not None:
        return np.clip(result, vrange[0], vrange[1], out=result)
    return result


def unsharp_mask(image, radius=1.0, amount=1.0, multichannel=False, preserve_range=False, sigma_color=0.25,
                 sigma_spatial=10, filter_='bilateral'):
    """
    Apply unsharp mask filter to the input image.
    Parameters:
    - image: ndarray
        Input image.
    - radius: float, optional
        Radius of the Gaussian kernel used for blurring the image. Default is 1.0.
    - amount: float, optional
        Amount of sharpening to apply to the image. Default is 1.0.
    - multichannel: bool, optional
        Whether the input image is multichannel (True) or not (False). Default is False.
    - preserve_range: bool, optional
        Whether to preserve the range of the input image or not. Default is False.
    - sigma_color: float, optional
        Standard deviation of the color space for the bilateral filter. Default is 0.25.
    - sigma_spatial: float, optional
        Standard deviation of the spatial space for the bilateral filter. Default is 10.
    - filter_: str, optional
        Type of filter to use. Default is 'bilateral'.
    Returns:
    - result: ndarray
        Filtered image.
    """

    vrange = None  # Range for valid values; used for clipping.
    if preserve_range:
        fimg = image.astype(np.float)
    else:
        fimg = img_as_float(image)
        negative = np.any(fimg < 0)
        if negative:
            vrange = [-1., 1.]
        else:
            vrange = [0., 1.]

    if multichannel:
        result = np.empty_like(fimg, dtype=np.float)
        for channel in range(image.shape[-1]):
            result[..., channel] = unsharp_mask_filter(
                fimg[..., channel], radius, amount, vrange,sigma_color,sigma_spatial,filter_)
        return result
    else:
        return unsharp_mask_filter(fimg, radius, amount, vrange,sigma_color,sigma_spatial,filter_='bilateral')


def _clean_segmask(seg, comp_id):
    """
    Takes in a segmentation and a specific structure id,
    finds the largest segmentation group within the entire mask and returns it.
    Input: image array
    return: largest segment
    """

    mask = 1 * (seg == comp_id) # makes everywhere with the compartment of interest equal to 1
    new_seg = None
    if any(np.unique(mask)) > 0:
        largestcomponent = _get_largest_component(seg=mask)
        new_seg = comp_id * largestcomponent

    return new_seg

def _get_largest_component(seg):
    """Takes in a segmentation and returns the largest connected component."""

    labels_array = label(seg)
    # take the bin count of all compartments excluding background to get the biggest size compartment, and then add 1 because the compartment label is one greater than its index
    largestcomponent = labels_array == np.argmax(np.bincount(labels_array.flat)[1:]) + 1

    return largestcomponent


def color_to_cat(label_array, color_list):
    """
    Changes labels to categorical for semantic segmentation targets 
    input: array of ground truth labels, dim: (nb images, nrow, ncol, nb channel=3),
        number of regions of interest/segments (including background)
    returns: categorical array of labels, dim: (nb of images, nrow, ncol, nb seg)
    """

    nb_img, h, w, _ = label_array.shape
    nb_seg = len(color_list)
    cat_array = np.zeros((nb_img, h, w, nb_seg))

    for i in range(nb_img):
        im = label_array[i]
        vectors = []
        for color in color_list:

            # array for each color, i.e. one-hot-encoded color arrays
            v = np.zeros((h, w), dtype=float)

            # getting indices for each color
            idx = np.where((im[:,:,0] == color[0]) & (im[:,:,1] == color[1]) & (im[:,:,2] == color[2])) 

            # set to 1 in corresponding vector array
            v[idx] = 1.0
            vectors.append(v)

        label = np.stack(vectors, axis=-1)
        cat_array[i] = label
    return cat_array


def get_alphashape(img_label, alpha_=3):
    """
    Generates an alpha shape mask based on the input image label.
    Parameters:
    img_label (ndarray): The input image label.
    alpha_ (float): The alpha value for the alpha shape.
    Returns:
    ndarray: The alpha shape mask.
    """

    img_contours = morph.dilation(img_label, morph.disk(2)) - morph.erosion(img_label,morph.disk(1))
    coords = corner_peaks(corner_harris(img_contours), min_distance=5, threshold_rel=0.0001) #0
    points = np.vstack([coords[:,1],coords[:,0]]).T

    alpha_shape = alphashape.alphashape(points*0.01, alpha_)

    alpha_coords0 = np.array(alpha_shape.exterior.coords)[:,0] * 100
    alpha_coords1 = np.array(alpha_shape.exterior.coords)[:,1] * 100
    
    r_mask = np.zeros(img_label.shape, dtype=np.double)

    rr, cc = polygon(alpha_coords1, alpha_coords0, img_label.shape)
    r_mask[rr, cc] = 1
    
    return r_mask.astype(int) #, alpha_coords0, alpha_coords1, points


def convert_a2c_to_a4c(a2c_cat):
    """
    Convert a 2-channel categorical mask to a 4-channel categorical mask.
    Args:
        a2c_cat (numpy.ndarray): The 2-channel categorical mask.
    Returns:
        numpy.ndarray: The 4-channel categorical mask.
    Raises:
        ValueError: If no right and left ventricle or no right and left atrium is detected.
    """

    # Add two empty channels for the right atrium and right ventricle
    a4c_cat = np.zeros((a2c_cat[0].shape[0], a2c_cat[0].shape[1], 5), dtype=np.uint8)
    for i in range(1, 3):
        mask = a2c_cat[0][..., i]

        label_image = label(mask)
        props = regionprops_table(label_image, properties=('label', 'bbox','area', 'filled_area', 'convex_area', 'centroid','eccentricity',
                                                            'orientation','major_axis_length','minor_axis_length', 'solidity'))
        props_df = pd.DataFrame(props)

        props_df.sort_values(by='area', ascending=False, inplace=True)
        props_df = props_df.iloc[:2]

        right_centroid = props_df['centroid-1'].min()
        left_centroid = props_df['centroid-1'].max()

        if right_centroid == left_centroid:
            if i == 1:
                raise ValueError('Error: No right and left ventricle detected')
            else:
                raise ValueError('Error: No right and left atrium detected')

        right_label = props_df[props_df['centroid-1'] == right_centroid]['label'].values[0]
        left_label = props_df[props_df['centroid-1'] == left_centroid]['label'].values[0]

        right_mask = np.zeros_like(label_image, dtype=np.uint8)
        right_mask[label_image == right_label] = 1
        left_mask = np.zeros_like(label_image, dtype=np.uint8)
        left_mask[label_image == left_label] = 1

        if i == 1:
            a4c_cat[..., 1] = left_mask
            a4c_cat[..., 3] = right_mask

        else:
            a4c_cat[..., 2] = left_mask
            a4c_cat[..., 4] = right_mask

    return a4c_cat


def stretch_rv(df, base_from):
    """
    Stretch the RV (Right Ventricle) in the given DataFrame based on the ratio of RV_L_seg to LV_L_seg.
    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the data.
    - base_from (str): The base directory path.
    Returns:
    None
    Raises:
    None
    Notes:
    - This function modifies the images in the given DataFrame by stretching the RV region based on the ratio of RV_L_seg to LV_L_seg.
    - The modified images are saved back to their original file paths.
    """

    stretched_folder = os.path.join(os.path.dirname(base_from), 'a4c_step1_labels_stretched_rv')
    if not os.path.exists(stretched_folder):
        os.makedirs(stretched_folder)

    for index, row in df.iterrows():

        if row['RV_L_seg'] == 0 or row['LV_L_seg'] == 0:
            continue

        ratio = row['RV_L_seg'] / row['LV_L_seg']

        if ratio < 0.8:

            filepath = os.path.join(base_from, row['Image'])

            new_length = row['RV_L_seg'] * (1 + (row['RV_L_seg'] / row['LV_L_seg']))

            stretch_factor = new_length / row['RV_L_seg']

            # Open the numpy file
            img_fov = np.load(filepath)

            contours, _ = cv2.findContours(img_fov[..., 4], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            x, y, w, h = cv2.boundingRect(contours[0])

            # Extract the region of interest (ROI)
            roi = img_fov[..., 4][y:y+h, x:x+w]

            new_height = int(h * stretch_factor)

            while y + h - new_height < 0:
                new_height -= 1

            # Resize the ROI (scale it upwards)
            scaled_roi = cv2.resize(roi, (w, new_height), interpolation=cv2.INTER_NEAREST)

            # Create a new mask with the same size as the original binary mask
            new_mask = np.zeros_like(img_fov[..., 4])

            # Calculate the position to place the scaled object
            bottom_y = y + h
            top_y = bottom_y - new_height

            new_mask[top_y:bottom_y, x:x+w] = scaled_roi

            img_fov[..., 4] = new_mask

        else:
            filepath = os.path.join(base_from, row['Image'])

            # Open the numpy file
            img_fov = np.load(filepath)

        new_filepath = os.path.join(stretched_folder, row['Image'])

        # Save the new image
        np.save(new_filepath, img_fov)
        print(f'Stretched RV in {row["Image"]}')


def get_mask_convex(img_filter):
    """
    Apply convex hull to create a mask. This is import to help with ws boundaries during flooding process of the watershed. 
    Input: 2D image
    Output: convex mask
    """
    img = img_filter.copy()
    img = pie_mask(img)
    l2 = label(morph.erosion(img>0.1,morph.disk(3))).astype(int)

    l2 = morph.remove_small_objects(l2, 350)
    mask_convex = morph.erosion(morph.convex_hull_image(l2).astype(int),morph.disk(10))
    img= img * mask_convex + (1- mask_convex) * 2
    
    return img


def pie_mask(img,c_x=0.5,c_y=0.9,a1=40,a2=140):
    """
    Create a pieslice mask to remove text, and other information outside of the scanning sector

    input: image array
    returns: pie mask array
    """

    xx0=int(img.shape[0]*(-1)*c_x)
    xx1=int((img.shape[1]+img.shape[0]*c_x))
    yy0= int((img.shape[0])*(-1)*c_y)
    yy1= int(img.shape[0]-10)
    mask = Image.new("L", (img.shape[1],img.shape[0]), 0)
    draw = ImageDraw.Draw(mask)
    draw.pieslice((xx0, yy0,xx1 , yy1), a1, a2, fill=1)  # draw an arc in black

    crop = img*mask
    return crop

def draw_circles(img, circles):
    """
    Draws circles on the input image.
    Parameters:
    - img: numpy.ndarray
        The input image.
    - circles: numpy.ndarray
        The array of circles to be drawn.
    Returns:
    - numpy.ndarray
        The image with circles drawn on it.
    """

    cimg = cv2.cvtColor((img * 255).astype('uint8'),cv2.COLOR_GRAY2BGR)
    cimg_z = np.zeros_like(cimg)

    for i in circles[0,:]:
    # draw the outer circle
        cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
        cv2.circle(cimg_z,(i[0],i[1]),i[2],(0,255,0),5)
        # draw the center of the circle
        cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
        cv2.putText(cimg,str(i[0])+str(',')+str(i[1]), (i[0],i[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255)
    return cimg_z

def detect_circles(img, minDist, minRadius=30, maxRadius=100):
    """
    Input: 
        img: 2D array image
        minDist: minimum distance between the centers of the detected circles. Set distance between circles to image shape if want to find only one circle in image.
        minRadius: minimum circle radius. Minimum radius and max radius are set by examining the image.
        maxRadius: maximum circle radius. Minimum radius and max radius are set by examining the image.
    output: image with the circle. 
    
    Prepare image to HoughCircle function. Try to reduce noise, and simplify the image before apply HoughCircles function.
    Some combination of blurring/filtering/canny is good for this.
    """
    
    img = (img/img.max() * 255).astype('uint8')
    gray_blur = cv2.medianBlur((img * 255).astype('uint8'), 11)  # Remove noise before laplacian
    gray_lap = cv2.Laplacian(gray_blur, cv2.CV_8UC1, ksize=5)  # Apply some edge detection algorithm. It can be Canny as well. 
    dilate_lap = cv2.dilate(gray_lap, (3, 3))
    lap_blur = cv2.bilateralFilter(dilate_lap, 9, 75, 75)
    
    ## Find circles (x,y,radius) 
    circles = cv2.HoughCircles(lap_blur, cv2.HOUGH_GRADIENT, dp=16, minDist=minDist, param1 = 100, param2=0.9, minRadius=minRadius, maxRadius=maxRadius)
    
    ## Draw circles using circle vector (x,y,radius) 
    img_circle = draw_circles((img).astype('uint8'), circles.astype(int))

    print("{} circles detected.".format(circles[0].shape[0]))
    return lap_blur, img_circle
