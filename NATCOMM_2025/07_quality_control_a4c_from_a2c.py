"""
This script runs quality control on A4C view measurements. Stretched LV when needed and saves file back to original path.

Usage: python 6_quality_control_a4c_from_a2c.py --indir /path/to/base/a4c_step1_labels
"""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import os

from utils import util_measurement, util_seg


def run_a4c_qc(indir):
    """
    Runs quality control on A4C view measurements. Stretched LV when needed.
    Controls for RV, LV, LA, and RA area and eccentricity.

    Args:
        indir (str): The base directory path containing the step1 A4C labels from A2C model.
    Returns:
        None
    Raises:
        None
    """

    pre_stretch_df = util_measurement.gen_qc_measures(indir, view='A4C')

    # Apply RV stretch
    util_seg.stretch_rv(pre_stretch_df, indir)

    # Re-calculate measurements
    post_stretch_df = util_measurement.gen_qc_measures(indir, view='A4C')

    # Apply qc thresholds
    post_stretch_df = post_stretch_df[(post_stretch_df['LV_A_seg'] > 4.7) & (post_stretch_df['LV_A_seg'] < 104.0)]    # Drop rows with incorrect lv length
    post_stretch_df = post_stretch_df[(post_stretch_df['RV_A_seg'] > 4.7) & (post_stretch_df['RV_A_seg'] < 104.0)]    # Drop rows with incorrect rv length
    post_stretch_df = post_stretch_df[(post_stretch_df['LA_A_seg'] > 6.0) & (post_stretch_df['LA_A_seg'] < 75.0)]    # Drop rows with incorrect la length
    post_stretch_df = post_stretch_df[(post_stretch_df['RA_A_seg'] > 6.0) & (post_stretch_df['RA_A_seg'] < 75.0)]    # Drop rows with incorrect ra length
    post_stretch_df = post_stretch_df[(post_stretch_df['ecc_LV'] > 0.62) & (post_stretch_df['ecc_LV'] < 0.96)]   # Drop rows with incorrect lv eccentricity
    post_stretch_df = post_stretch_df[(post_stretch_df['ecc_RV'] > 0.65) & (post_stretch_df['ecc_RV'] < 0.96)]   # Drop rows with incorrect rv eccentricity
    post_stretch_df = post_stretch_df[(post_stretch_df['ecc_LA'] > 0.3) & (post_stretch_df['ecc_LA'] < 0.96)]   # Drop rows with incorrect la eccentricity
    post_stretch_df = post_stretch_df[(post_stretch_df['ecc_RA'] > 0.17) & (post_stretch_df['ecc_RA'] < 0.95)]   # Drop rows with incorrect ra eccentricity

    print(f'Number of examples before QC: {len(pre_stretch_df)}')
    print(f'Number of examples after QC: {len(post_stretch_df)}')

    post_stretch_df.to_csv(os.path.join(os.path.dirname(indir), 'a4c_step1_labels_passed_qc.csv'), index=False)
    print('A4C QC df saved to a4c_step1_labels.csv')


if __name__ == '__main__':

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--indir', type=str, help='Base directory containing the A4C step 1 labels.')
    args = parser.parse_args()

    run_a4c_qc(args.indir)
