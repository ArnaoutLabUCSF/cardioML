"""
This file generates watershed labels for images in the specified directory.

Usage: python 1_generate_watershed_labels.py --indir /path/to/a2c_images
"""

import numpy as np
from glob import glob
import os
import cv2
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from utils.util_seg import unsharp_mask, get_label_ws


def generate_watershed_labels(indir: str):
    """
    Generate watershed labels for images in the specified directory.
    Args:
        indir (str): The base directory path.
    Returns:
        None
    """

    filenames = glob(os.path.join(indir, '*.npy'))

    watershed_dir = os.path.join(os.path.dirname(indir), 'a2c_watershed_labels')

    if not os.path.exists(watershed_dir):
        os.makedirs(watershed_dir)

    for filename in filenames:
        watershed_path = os.path.join(watershed_dir, os.path.basename(filename))

        if os.path.exists(watershed_path):
            continue

        img_fov = np.load(filename)
        img_256 = cv2.resize(img_fov, (256, 256), interpolation=cv2.INTER_AREA)
        img_normalized = ((img_256 - img_256.min()) / (img_256.max() - img_256.min())).astype('float32')
        img_gray = img_normalized[..., 0]

        img_bi = unsharp_mask(img_gray, 20, 20)

        try:
            _, _, _, img_prop_cat, _ = get_label_ws(img_bi, 20, c=20, l=150, r=260)
        except:
            print(f'Could not generate watershed label for {filename}')
            continue

        np.save(watershed_path, img_prop_cat)


## Main Code

if __name__ == '__main__':

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--indir', type=str, help='Directory containing a2c images')
    args = parser.parse_args()

    generate_watershed_labels(args.indir)
