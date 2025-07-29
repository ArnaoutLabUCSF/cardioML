"""
Usually we would run this file to generate filled hed predictions from our data using our trained sax hed model.
Because our example hed model is trained on very little data, and is therefore too weak, we will use pre-generated predictions from our real hed model in the steps following.
"""

import numpy as np
from glob import glob
from skimage.morphology import label, disk, dilation
from skimage import measure
from scipy import ndimage as ndi
import cv2
import os
from tqdm import tqdm
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from utils import util_seg
from model_arch import hed


def generate_hed_labels(indir, model):
    """
    Generate HED labels for images in a directory using a given model.
    Args:
        indir (str): The base directory path where the images are located.
        model: The model used for predicting HED labels.
    Returns:
        None
    """

    filenames = glob(os.path.join(indir, '*.npy'))

    with tqdm(total=len(filenames)) as pbar:

        for filename in filenames:

            filled_folder = os.path.join(os.path.dirname(indir), 'sax_filled_hed_labels')
            if not os.path.exists(filled_folder):
                os.makedirs(filled_folder)

            hed_pred_folder = os.path.join(os.path.dirname(indir), 'sax_hed_preds')
            if not os.path.exists(hed_pred_folder):
                os.makedirs(hed_pred_folder)

            img_shapes_folder = os.path.join(os.path.dirname(indir), 'sax_shapes')
            if not os.path.exists(img_shapes_folder):
                os.makedirs(img_shapes_folder)

            largest_comp_folder = os.path.join(os.path.dirname(indir), 'sax_filled_masks')
            if not os.path.exists(largest_comp_folder):
                os.makedirs(largest_comp_folder)

            filled_path = os.path.join(filled_folder, os.path.basename(filename))
            hed_pred_path = os.path.join(hed_pred_folder, os.path.basename(filename))
            img_shapes_path = os.path.join(img_shapes_folder, os.path.basename(filename))
            largest_comp_path = os.path.join(largest_comp_folder, os.path.basename(filename))

            if os.path.exists(filled_path):
                pbar.update(1)
                continue

            img_fov = np.load(filename)

            img_480 = cv2.resize(img_fov, (480,480), interpolation = cv2.INTER_AREA)
            img_480 = (img_480 - img_480.min()) / (img_480.max() - img_480.min())

            pred_hed = model.predict(img_480[np.newaxis])       

            img_hed = (pred_hed[5][0,...,0]/pred_hed[5][0,...,0].max())

            img_label = label(1-(img_hed > 0.6))
            img_shapes = np.zeros_like(img_label)
            number_labels = len(np.unique(img_label))

            color_list_axis_mid = [(0,0,0), (0,0,0), (255,0,0)] #, (0,128,255) #(0,0,0)

            for i in range(number_labels):
                ### Find the size of each segment in the image so we can exclude the outer area of the images.
                img_fill = ndi.binary_fill_holes(dilation((img_label == i).astype(int),disk(2))).astype(int)
                regions = measure.regionprops(img_fill)
                ecc = regions[0].eccentricity

                if img_fill.sum() > 2000 and img_fill.sum() < 25000 and ecc < 0.88:
                    img_shapes = img_shapes + img_fill

            try:
                filled_mask = dilation(util_seg._get_largest_component(img_shapes), disk(2))
            except:
                print('Could not generate filled mask for {i} in', filename)
                pbar.update(1)
                continue

            img_3ch = (cv2.cvtColor(label(filled_mask).astype('float32'), cv2.COLOR_BGR2RGB) * 255).astype('uint8')
            img_rgb = util_seg.cat_to_color(img_3ch[np.newaxis].astype(int), color_list_axis_mid)

            np.save(filled_path, img_rgb[0])
            np.save(hed_pred_path, (img_hed>0.6).astype(int)+(img_480[...,0]))
            np.save(img_shapes_path, img_shapes)
            np.save(largest_comp_path, filled_mask)

            pbar.update(1)


if __name__ == '__main__':

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--indir', type=str, help='Base directory containing the SAX images.')
    parser.add_argument('--weights', type=str, help='Path to weights file')
    args = parser.parse_args()

    # Load the HED model
    model = hed.hed()

    model.load_weights(args.weights)

    generate_hed_labels(args.indir, model)
