from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Activation, Add, BatchNormalization
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Cropping2D


def create_resnet(input_dim, nb_classes, dropout):
	"""
	CNN model based on ResNet
	input: input dimension, 
		number of classes,
		list of dropout values
	"""
	inputs = Input(shape=(input_dim))

	# MODULE 1
	conv1 = Conv2D(32, kernel_size=(3,3), strides=(1,1), activation='linear', padding='same', use_bias=False)(inputs)
	conv1 = BatchNormalization(epsilon=1.001e-5)(conv1)
	conv1 = Activation('relu')(conv1)

	# MODULE 2
	conv2 = Conv2D(64, kernel_size=(3,3), strides=(2,2), activation='linear', use_bias=False)(conv1)
	conv2_a = BatchNormalization(epsilon=1.001e-5)(conv2)
	conv2_a = Activation('relu')(conv2_a)

	conv3 = Conv2D(64, kernel_size=(3,3), strides=(1,1), activation='linear', padding='same', use_bias=False)(conv2_a)
	conv3 = BatchNormalization(epsilon=1.001e-5)(conv3)
	conv_skip1 = Add()([conv2, conv3])
	conv_skip1 = Activation('relu')(conv_skip1)

	# MODULE 3
	conv4 = Conv2D(128, kernel_size=(3,3), strides=(2,2), activation='linear', use_bias=False)(conv_skip1)
	conv4_a = BatchNormalization(epsilon=1.001e-5)(conv4)
	conv4_a = Activation('relu')(conv4_a)

	conv5 = Conv2D(128, kernel_size=(3,3), strides=(1,1), activation='linear', padding='same', use_bias=False)(conv4_a)
	conv5 = BatchNormalization(epsilon=1.001e-5)(conv5)
	conv_skip2 = Add()([conv4, conv5]) 
	conv_skip2 = Activation('relu')(conv_skip2)

	# MODULE 4
	conv6 = Conv2D(256, kernel_size=(3,3), strides=(2,2), activation='linear', use_bias=False)(conv_skip2)
	conv6_a = BatchNormalization(epsilon=1.001e-5)(conv6)
	conv6_a = Activation('relu')(conv6_a)
    
	conv7 = Conv2D(256, kernel_size=(3,3), strides=(1,1), activation='linear', padding='same', use_bias=False)(conv6_a)
	conv7 = BatchNormalization(epsilon=1.001e-5)(conv7)
	conv_skip3 = Add()([conv6, conv7])
	conv_skip3 = Activation('relu')(conv_skip3)

	# MODULE 5
	final = GlobalAveragePooling2D()(conv_skip3)
	final = Dense(512, use_bias=False)(final)
	final = BatchNormalization(epsilon=1.001e-5)(final)
	final = Activation('relu')(final)
	final = Dropout(dropout)(final)
	predictions = Dense(nb_classes, activation='softmax')(final)

	model = Model(inputs=[inputs], outputs=[predictions])
	return model


def create_unet8(input_dim, nb_classes, dropout_list):
	"""
	Unet model based on paper 
		"U-Net: Convolutional Networks for Biomedical Image Segmentation"
	input: input dimension, number of classes, dropout list
	"""
	inputs = Input(shape=(input_dim))

	conv1 = Conv2D(8, (3,3), activation='relu', padding='same')(inputs)
	conv1 = Conv2D(8, (3,3), activation='relu', padding='same')(conv1)
	pool1 = MaxPooling2D(pool_size=(2,2))(conv1)

	conv2 = Conv2D(16, (3,3), activation='relu', padding='same')(pool1)
	conv2 = Conv2D(16, (3,3), activation='relu', padding='same')(conv2)
	pool2 = MaxPooling2D(pool_size=(2,2))(conv2)

	conv3 = Conv2D(32, (3,3), activation='relu', padding='same')(pool2)
	conv3 = Conv2D(32, (3,3), activation='relu', padding='same')(conv3)
	pool3 = MaxPooling2D(pool_size=(2,2))(conv3)

	conv4 = Conv2D(64, (3,3), activation='relu', padding='same')(pool3)
	conv4 = Conv2D(64, (3,3), activation='relu', padding='same')(conv4)
	pool4 = MaxPooling2D(pool_size=(2,2))(conv4)

	conv5 = Conv2D(128, (3,3), activation='relu', padding='same')(pool4)
	conv5 = Conv2D(128, (3,3), activation='relu', padding='same')(conv5)


	upconv1 = Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(conv5)
	concat1 = concatenate([upconv1, conv4])
	concat1 = Dropout(dropout_list[0])(concat1)
	conv6 = Conv2D(64, (3,3), activation='relu', padding='same')(concat1)
	conv6 = Conv2D(64, (3,3), activation='relu', padding='same')(conv6)

	upconv2 = Conv2DTranspose(32, (2,2), strides=(2,2), padding='same')(conv6)
	concat2 = concatenate([upconv2, conv3]) 
	concat2 = Dropout(dropout_list[1])(concat2)
	conv7 = Conv2D(32, (3,3), activation='relu', padding='same')(concat2)
	conv7 = Conv2D(32, (3,3), activation='relu', padding='same')(conv7)

	upconv3 = Conv2DTranspose(16, (2,2), strides=(2,2), padding='same')(conv7)
	concat3 = concatenate([upconv3, conv2]) 
	concat3 = Dropout(dropout_list[2])(concat3)
	conv8 = Conv2D(16, (3,3), activation='relu', padding='same')(concat3)
	conv8 = Conv2D(16, (3,3), activation='relu', padding='same')(conv8)

	upconv4 = Conv2DTranspose(8, (2,2), strides=(2,2), padding='same')(conv8)
	concat4 = concatenate([upconv4, conv1]) 
	concat4 = Dropout(dropout_list[3])(concat4)
	conv9 = Conv2D(8, (3,3), activation='relu', padding='same')(concat4)
	conv9 = Conv2D(8, (3,3), activation='relu', padding='same')(conv9)
	predictions = Conv2D(nb_classes, (1,1), activation='softmax')(conv9)

	model = Model(inputs=[inputs], outputs=[predictions])
	return model