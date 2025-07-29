import os
from argparse import ArgumentParser, RawDescriptionHelpFormatter
import pandas as pd
from utils import util_dataframes, util_measurement, FittingSinusoid
from model_arch import unet
from glob import glob

def load_models(path_to_model: str = 'real_models/'):

    a2cmodel = unet.get_unet(256, 256, img_ch=3, ch=3)
    a4cmodel = unet.get_unet(256, 256, img_ch=3, ch=5)
    saxmodel = unet.get_unet(256, 256, img_ch=1, ch=3)

    a2cmodel.load_weights(f'{path_to_model}2ndRound_UNet_A2CModel_April2019_SelfLearning_MaskAug.h5')

    a4cmodel.load_weights(f'{path_to_model}2ndRound_UNet_A4CModel_040517Dataset.h5')

    saxmodel.load_weights(f'{path_to_model}step4_unet_label_endocardial_epicardial_seg.h5')

    return a2cmodel, a4cmodel, saxmodel

def generate_measurements_df(indir, a2cmodel, a4cmodel, saxmodel):

    files = glob(f'{indir}/**/*.npy', recursive=True)
    subdirs = list(set([os.path.dirname(f) for f in files]))

    for subdir in subdirs:

        # Load model --> ch=5 for a4c model, and ch=3 for a2c model
        view = subdir.split('/')[-1]
        print('---- >>>> View is:', view)

        if os.path.exists(f"{os.path.dirname(subdir)}/{view}_measures_R2.csv"):
            continue

        ## Create dataframe of filenames
        df_filenames = util_dataframes.dataframe_of_filenames(subdir, path_to='False', file_type='npy')
        filenames = df_filenames.fn.sort_values().values

        ## Compute measurements
        filename_csv = f"{os.path.dirname(subdir)}/measures_{view}.csv"
        if view == 'A2C':
            df_measurement = util_measurement.gen_df_measurements(filenames, filename_csv, a2cmodel, view)
        elif view == 'A4C':
            df_measurement = util_measurement.gen_df_measurements(filenames, filename_csv, a4cmodel, view)
        elif view == 'SAXMID':
            df_measurement = util_measurement.gen_df_measurements(filenames, filename_csv, saxmodel, view)


        # ## Merge RR
        rr_df = pd.read_csv(f'{indir}/ID_clip_RR.csv')
        df_measurement = df_measurement.merge(rr_df, on='ID_clip', how='left')
        df_measurement.to_csv(filename_csv, index=False)

        # Fit sinusoid and compute Rsquare
        df_measurement = df_measurement.sort_values(['ID_clip','frame'])

        ## Fix columns order
        cols = df_measurement.columns.tolist()
        cols = cols[-7:] + cols[:-7]
        df_measurement = df_measurement[cols]

        df_result = pd.DataFrame([])
        for label_ in df_measurement.label.unique():
            print("    Fitting Sinusoid: %s / %s" %(label_, len(df_measurement.label.unique())), end="\r")
            print("    Fitting Sinusoid:" )

            df_temp = df_measurement[df_measurement['label']==label_]
            df_rsquare = FittingSinusoid.computeR2(df_temp, False)
            df_temp = df_temp.merge(df_rsquare,on=['anonid', 'ID_clip'], how='left')
            df_result = df_result.append(df_temp)
        df_result.reset_index(False, True)

        filename_csv = f"{os.path.dirname(subdir)}/{view}_measures_R2.csv"

        df_result.to_csv(filename_csv, index=False)


if __name__ == "__main__":
    parser = ArgumentParser(description="Compute measurements from A2C and A4C chambers. Then fit sinusoid and compute R2 between fitting and data",
    formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("--indir", type=str, help="Directory containing dicom files.")
    parser.add_argument("--path_to_models", type=str, help="Path to model weights.")
    args = parser.parse_args()

    if args.path_to_models:
        a2cmodel, a4cmodel, saxmodel = load_models(args.path_to_models)
    else:
        a2cmodel, a4cmodel, saxmodel = load_models()

    generate_measurements_df(args.indir, a2cmodel, a4cmodel, saxmodel)
