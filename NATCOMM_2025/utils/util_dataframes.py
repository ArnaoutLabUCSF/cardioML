import pandas as pd
from pathlib import Path
        
def dataframe_of_filenames(indir, path_to=None, save = False, file_type='npy'):
    '''
    It generates a dataframe containing image path, ID and ID_clip.
    Input: path to label files. File types can be npy, png, and npy.
    Output: dataframe of labels info
    '''
    print('Generating DataFrame ...')
    print()
    images_path = Path(indir)
    filenames = list(images_path.glob(f'**/*.{file_type}'))

    info = []
    j=0
    for f in filenames:
        print("    Status: %s / %s" %(j, len(filenames)), end="\r")
        j+=1
        if file_type == 'npy' or file_type == 'png':
            info.append({'fn':str(f),'ImgName':str(f.name),'anonid':str(f.name).split('_',2)[0], 'ID_clip':str(f.name).split('_',2)[0] + '_' + str(f.name).split('_',2)[1],
                        'frame':str(f.name).split('_')[-1].split('.')[0]}) # , 'nbFrames':str(f.name).split('_',4)[4].split('.')[0]}) 
        elif file_type == 'dcm':
            info.append({'fn':str(f),'ImgName':str(f.name),'anonid':str(f.name).split('_',2)[0], 'ID_clip':str(f.name).split('_',2)[0] + '_' + str(f.name).split('_',2)[1].replace('Image-','')
                        })
        else:
            print('Error in image format.')
        
    df = pd.DataFrame(info)
    # print(f'Number of images found {df.shape[0]} from {len(df.anonid.unique())} unique IDs, and {len(df.ID_clip.unique())} ID_clips.')
    
    if save:
        df.to_csv(f'{path_to}', mode='w', sep=',', encoding='utf8', header=True, index=False)
    return df
