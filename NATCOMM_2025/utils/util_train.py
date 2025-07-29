import albumentations as A
import keras
import numpy as np
from scipy import ndimage as ndi
import random
import cv2
import os

from utils import util_seg


def padwithzero_square_shape(img,shape=None):
    """
    input: image array
    returns: square padded image
    """
    L = max(img.shape[0],img.shape[1])
    if shape:
        L = shape
    width_new = L
    height_new = L
    img = img[0:width_new,0:height_new,:]
    delta_w = abs(width_new - img.shape[1] )
    delta_h = abs(height_new - img.shape[0] )
    top, bottom = delta_h // 2, delta_h - (delta_h // 2) 
    left, right = delta_w // 2, delta_w - (delta_w // 2) 
    return np.pad(img, ((top, bottom), (left, right),(0,0)), mode='constant',constant_values=(0))


def transform(image, mask):
    """
    Applies a series of image augmentation techniques to the input image and mask.
    Args:
        image (numpy.ndarray): The input image.
        mask (numpy.ndarray): The corresponding mask.
    Returns:
        numpy.ndarray: The augmented image.
        numpy.ndarray: The augmented mask.
    """

    aug = A.Compose([
            A.Crop(x_min=30, x_max=220, y_min=30, y_max=170, p = 0.05),
            A.PadIfNeeded(min_height=256, min_width=256, border_mode=0, value=0, mask_value=0, p=1),
            #A.CropAndPad(percent = [-0.2,0], sample_independently=False, p = 0.05),
            A.HorizontalFlip(p=0.05),
            A.Downscale(0.25,0.8, p=0.05), #Decreases image quality by downscaling and upscaling back
            A.PixelDropout(mask_drop_value = None, drop_value = None, p=0.05),
            A.Affine(scale=(0.8,1.1),keep_ratio=True, translate_percent = (-0.01, 0.01), rotate=(-10,10),p=0.1),
            A.OneOf([
            A.MedianBlur(),
            A.GaussNoise(),
            A.AdvancedBlur(),
        ], p=0.2),
        A.OneOf([
            A.MedianBlur(),
            A.GaussNoise(),
            A.AdvancedBlur(),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=40.0), #Contrast Limited Adaptive Histogram Equalization
            A.ColorJitter(), #Randomly changes the brightness, contrast, and saturation of an image.
            A.RandomToneCurve(), #Randomly change the relationship between bright and dark areas of the image by manipulating its tone curve.
            A.Sharpen(p=0.8),
            A.RandomBrightnessContrast(p=0.8),    
            A.RandomGamma(p=0.8)
        ], p=0.2),
        ], p = 0.5)
    image = np.uint8(image * 255)
    augmented = aug(image=image, mask=mask)
    augmented_image = augmented['image']
    augmented_image = (augmented_image - augmented_image.min()) / (augmented_image.max() - augmented_image.min())
    augmented_mask = augmented['mask'].astype('int')

    return augmented_image.astype('float32'), augmented_mask


def aug_mask(img_echo, img_label, ch=2):
    """
    Applies augmentation to the input image and label.
    Args:
        img_echo (ndarray): The input image.
        img_label (ndarray): The input label.
        ch (int, optional): The channel index to use for finding objects. Defaults to 2.
    Returns:
        tuple: A tuple containing the augmented image and label.
    Raises:
        None
    Examples:
        >>> img_echo = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> img_label = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
        >>> aug_mask(img_echo, img_label)
        (array([[5, 6],
                [8, 9]]), array([[0, 1],
                                 [1, 0]]))
    """


    slice_x, slice_y = ndi.find_objects((img_label[...,ch]))[0]
    stop = slice_y.stop+30 if (slice_y.stop+30) < 256 else slice_y.stop
    k = random.randrange(20, 35, 1)
    echo_aug = img_echo[k:slice_x.stop+k, slice_y.start-k:stop]
    echo_aug = padwithzero_square_shape(echo_aug, max(echo_aug.shape))
    echo_aug = cv2.resize(echo_aug, (256,256), interpolation = cv2.INTER_AREA)

    label_aug = img_label[k:slice_x.stop+k, slice_y.start-k:stop]
    label_aug = padwithzero_square_shape(label_aug, max(label_aug.shape))
    label_aug = cv2.resize(label_aug.astype('float32'), (256,256), interpolation = cv2.INTER_AREA).astype(int)
    
    return echo_aug, label_aug.astype(int)


#Create a callback
class LossHistory(keras.callbacks.Callback):
    """
    A callback class to track and store the loss values during training.
    Attributes:
        losses (list): A list to store the loss values.
    Methods:
    on_train_begin(logs={}): Called at the beginning of training.
    on_batch_end(batch, logs={}): Called at the end of each batch.


    Called at the beginning of training.
    Args:
        logs (dict): Dictionary containing the training metrics.
    Returns:
        None

    Called at the end of each batch.
    Args:
        batch (int): The current batch index.
        logs (dict): Dictionary containing the training metrics.
    Returns:
        None
    """

    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


class DataGenerator(keras.utils.Sequence):
    """
    Custom data generator for training a model.
    Args:
        list_IDs (list): List of IDs for the data samples.
        labels (list): List of labels for the data samples.
        path_img (str): Path to the directory containing the image data.
        path_label (str): Path to the directory containing the label data.
        path_self_learning (str): Path to the directory containing the self-learning data.
        batch_size (int, optional): Number of samples per batch. Defaults to 4.
        dim (tuple, optional): Dimensions of the input data. Defaults to (360, 300).
        n_channels (int, optional): Number of channels in the input data. Defaults to 3.
        shuffle (bool, optional): Whether to shuffle the data samples. Defaults to True.
        augment (bool, optional): Whether to apply data augmentation. Defaults to False.
        model (object, optional): Pre-trained model object. Defaults to None.
        aug (object, optional): Augmentation object. Defaults to None.
        self_learning (bool, optional): Whether to use self-learning data. Defaults to False.
    Methods:
        __len__(): Returns the number of batches per epoch.
        __getitem__(index): Generates one batch of data.
        on_epoch_end(): Updates indexes after each epoch.
        __data_generation(list_IDs_temp): Generates data containing batch_size samples.
    """

    def __init__(self, list_IDs,  labels, path_img, path_label, path_self_learning,
                 batch_size=4, dim=(360,300), n_channels=3, shuffle=True, augment=False,
                 model=None, aug = None, self_learning=False, saxmid=False, saxunet=False):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.path_img = path_img
        self.path_label = path_label
        self.path_self_learning = path_self_learning
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.augment = augment
        self.model = model
        self.self_learning = self_learning
        self.on_epoch_end()
        self.aug = aug
        self.saxmid = saxmid
        self.saxunet = saxunet

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y
    
    # At the end of the epoch Keras calls this function. 
    # It will shuffle the samples for the next epoch.
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        if self.model != None:
            list_IDs_temp = [self.list_IDs[k] for k in self.indexes]
    
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        xdim = self.dim[0]
        ydim = self.dim[1]

        X = np.empty((self.batch_size, xdim, ydim, 3)) 
        y = np.empty((self.batch_size, xdim, ydim, self.n_channels)) 
        
        if not self.saxmid:
            for i, ID in enumerate(list_IDs_temp):

                    npy_X = np.load(os.path.join(self.path_img, ID)) 
                    npy_X = (npy_X - npy_X.min()) / (npy_X.max() - npy_X.min())

                    if not self.self_learning:
                        pred = np.load(os.path.join(self.path_label, ID))
                    else:
                        pred = np.load(os.path.join(self.path_self_learning, ID))
                    
                    npy_y = util_seg.refine_chambers(pred)

                    if self.augment:
                        ## Augument images
                        npy_X, npy_y = transform(image=npy_X, mask=npy_y.astype(int))
                        
                        if np.random.rand(1) > 0.8:
                            npy_X, npy_y = aug_mask(npy_X, npy_y.astype(int))
                                    
                    npy_X = (npy_X - npy_X.min()) / (npy_X.max() - npy_X.min())

                    X[i,] = npy_X[...].astype('float32')
                    y[i,] = npy_y.astype(np.uint8)

            return X.astype('float32'), y.astype('uint8')

        else:
            if not self.saxunet:
                for i, ID in enumerate(list_IDs_temp):

                    npy_X = np.load(f'{self.path_img}{ID}')
                    npy_X = (npy_X - npy_X.min()) / (npy_X.max() - npy_X.min())
                    
                    npy_y = np.load(f'{self.path_label}{ID}')
                    
                    X[i,] = npy_X.astype('float32')
                    y[i,] = npy_y[...,np.newaxis].astype(int)
                
                return np.array(X), [np.array(y), np.array(y), np.array(y), np.array(y), np.array(y), np.array(y)]
            else:
                for i, ID in enumerate(list_IDs_temp):
                    npy_X = np.load(f'{self.path_img}{ID}')            
                    npy_X = cv2.resize(npy_X, (256, 256), interpolation = cv2.INTER_AREA)
                    npy_X = (npy_X - npy_X.min()) / (npy_X.max() - npy_X.min())
                    
                    if not self.self_learning:
                        npy_y = np.load(f'{self.path_label}{ID}').astype('uint8')
                        npy_y = cv2.resize(npy_y, (256, 256), interpolation = cv2.INTER_AREA)
                        npy_y = (npy_y - npy_y.min()) / (npy_y.max() - npy_y.min())
                    else:
                        npy_y = np.load(f'{self.path_self_learning}{ID}').astype('uint8')
                    
                    ### Dataaugmentaion - random crop
                    if np.random.rand(1) > 0.5:
                        npy_X,npy_y = util_seg.random_crop(npy_X, npy_y, 128, 128)
                        npy_X = cv2.resize((npy_X*255).astype('uint8'), (256, 256), interpolation = cv2.INTER_AREA) / 255
                        npy_y = cv2.resize((npy_y*255).astype('uint8'), (256, 256), interpolation = cv2.INTER_AREA) / 255
                        

                    X[i,] = npy_X[...,0:1].astype('float32')
                    y[i,] = npy_y[...].astype('uint8')

                return np.array(X), np.array(y)
