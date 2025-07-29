"""
Runs quality control on watershed labels and saves a filtered csv file.
Controls for LV and LA area and eccentricity.

Usage: python 2_quality_control_watershed_labels.py --indir /path/to/watershed_labels --outdir /path/to/save/qc_csv
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from utils import util_measurement
import os


def run_watershed_qc(indir):
    """
    Runs quality control on watershed labels.
    Args:
        indir (str): The base directory path.
        outdir (str): The output directory path.
    Returns:
        None
    """

    df_measurements = util_measurement.gen_qc_measures(indir, view='A2C')
    df = df_measurements.copy()

    # Apply qc thresholds
    df = df[(df['LV_A_seg'] > 4.7) & (df['LV_A_seg'] < 104.0)]    # Drop rows with incorrect lv length
    df = df[(df['LA_A_seg'] > 6.0) & (df['LA_A_seg'] < 75.0)]    # Drop rows with incorrect la length
    df = df[(df['ecc_LV'] > 0.62) & (df['ecc_LV'] < 0.96)]   # Drop rows with incorrect lv eccentricity
    df = df[(df['ecc_LA'] > 0.3) & (df['ecc_LA'] < 0.96)]   # Drop rows with incorrect la eccentricity

    print(f'Number of examples before QC: {len(df_measurements)}')
    print(f'Number of examples after QC: {len(df)}')

    outdir = os.path.dirname(indir)

    df.to_csv(os.path.join(outdir, 'a2c_watershed_labels_passed_qc.csv'), index=False)
    print(f'QC csv saved to {outdir}')


if __name__ == '__main__':

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--indir', type=str, help='Directory containing watershed labels')
    args = parser.parse_args()

    run_watershed_qc(args.indir)
