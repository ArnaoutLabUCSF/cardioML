"""
This file generates hough circle labels for images in the specified directory.

Usage: python 9_generate_hough_circle_labels.py --indir /path/to/sax_images
"""

import numpy as np
from glob import glob
from tqdm import tqdm
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from utils import util_seg


def generate_hough_circles(indir):
    """
    Generate Hough circles for images in a directory.
    Saves hough cirlces to subdirectory 'sax_hough_circle_labels'.
    
    Args:
        indir (str): The input directory containing the images.
    Returns:
        None
    """

    filenames = glob(f'{indir}/*.npy')

    with tqdm(total=len(filenames)) as pbar:

        for filename in filenames:

            label_folder = os.path.join(os.path.dirname(indir), 'sax_hough_circle_labels')

            if not os.path.exists(label_folder):
                os.makedirs(label_folder)

            dest_filename = os.path.join(label_folder, os.path.basename(filename))

            if os.path.exists(dest_filename):
                pbar.update(1)
                continue

            img_fov = np.load(filename)
            _, img_circle = util_seg.detect_circles(util_seg.get_mask_convex(img_fov[...,0]), minDist=400, minRadius=30, maxRadius=100)

            np.save(dest_filename, img_circle[...,1])
            pbar.update(1)


### Main Code
if __name__ == '__main__':

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--indir', type=str, required=True, help='The input directory containing the SAX images.')
    args = parser.parse_args()

    generate_hough_circles(args.indir)
