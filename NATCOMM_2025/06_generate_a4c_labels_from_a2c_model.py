"""
This file generates A4C labels from A2C model.

Usage: python 5_generate_a4c_labels_from_a2c_model.py --weights /path/to/weights --indir /path/to/base
"""

from glob import glob
import numpy as np
import cv2
import os
from tqdm import tqdm
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from model_arch import unet
from utils import util_seg


def generate_a4c_labels(path_to_weights, indir):
    """
    Generate A4C labels from A2C model.
    Args:
        path_to_weights (str): Path to the weights of the A2C model.
        indir (str): Base directory containing the A4C images.
    Returns:
        None
    """

    modelA2C = unet.get_unet(img_rows=256, img_cols=256, img_ch=3, ch=3)

    modelA2C.load_weights(path_to_weights)

    a4c_filenames = glob(os.path.join(indir, '*.npy'))

    step1_label_dir = os.path.join(os.path.dirname(indir), 'a4c_step1_labels')

    if not os.path.exists(step1_label_dir):
        os.makedirs(step1_label_dir)

    with tqdm(total=len(a4c_filenames)) as pbar:
        for filename in a4c_filenames:

            step1_label_path = os.path.join(step1_label_dir, os.path.basename(filename))

            # If the label has already been generated, skip
            if os.path.exists(step1_label_path):
                pbar.update(1)
                continue

            img_fov = np.load(filename)
            img_256 = cv2.resize(img_fov, (256,256), interpolation = cv2.INTER_AREA)
            img_normalized = ((img_256-img_256.min())/(img_256.max() - img_256.min())).astype('float32')

            # Step 1 prediction
            a2c_cat = util_seg.get_img_pred(img_normalized, modelA2C, 'a2c')
            try:
                a4c_cat = util_seg.convert_a2c_to_a4c(a2c_cat)
            except:
                print(f'Could not generate a4c label for {filename}')
                pbar.update(1)
                continue

            a4c_cat = util_seg.refine_chambers(a4c_cat, view='A4C')

            np.save(step1_label_path, a4c_cat)

            pbar.update(1)


if __name__ == '__main__':

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--weights', type=str, help='Path to weights file')
    parser.add_argument('--indir', type=str, help='Base directory path')
    args = parser.parse_args()

    generate_a4c_labels(args.weights, args.indir)
