from vggUnet import VggUnet,loss_func,dice_coef
from HelperFunctions import getSliceFromImage
import nibabel as nib
import numpy as np
from random import sample,shuffle
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.layers.core import Lambda
import keras.backend as K

def attention_up_and_concate(down_layer, layer, data_format='channels_first'):
    merge_axis = -1
    if data_format == 'channels_first':
        in_channel = down_layer.get_shape().as_list()[1]
        merge_axis = 1
    else:
        in_channel = down_layer.get_shape().as_list()[3]

    # up = Conv2DTranspose(out_channel, [2, 2], strides=[2, 2])(down_layer)
    up = UpSampling2D(size=(2, 2), data_format=data_format)(down_layer)

    layer = attention_block_2d(x=layer, g=up, inter_channel=in_channel // 4, data_format=data_format)
    concate = concatenate([up, layer], axis = merge_axis)
    return concate


def attention_block_2d(x, g, inter_channel, data_format='channels_first'):
    theta_x = Conv2D(inter_channel, [1, 1], strides=[1, 1], data_format=data_format)(x)
    phi_g = Conv2D(inter_channel, [1, 1], strides=[1, 1], data_format=data_format)(g)
    f = Activation('relu')(add([theta_x, phi_g]))
    psi_f = Conv2D(1, [1, 1], strides=[1, 1], data_format=data_format)(f)
    rate = Activation('sigmoid')(psi_f)
    att_x = multiply([x, rate])
    return att_x

def att_unet(img_w, img_h, n_classes, data_format='channels_last'):
    IMAGE_ORDERING = data_format
    MERGE_AXIS = -1
    img_input = Input(shape=(img_w, img_h, 3))
    x = Conv2D(64, (3, 3), activation='relu', padding='same',
                   name='block1_conv1', data_format=IMAGE_ORDERING)(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same',
                name='block1_conv2', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool',
                        data_format=IMAGE_ORDERING)(x)
    f1 = x
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same',
                name='block2_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same',
                name='block2_conv2', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool',
                        data_format=IMAGE_ORDERING)(x)
    f2 = x

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same',
                name='block3_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same',
                name='block3_conv2', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same',
                name='block3_conv3', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool',
                        data_format=IMAGE_ORDERING)(x)
    f3 = x

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
                name='block4_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
                name='block4_conv2', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
                name='block4_conv3', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool',
                        data_format=IMAGE_ORDERING)(x)
    f4 = x

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
                name='block5_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
                name='block5_conv2', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
                name='block5_conv3', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool',
                        data_format=IMAGE_ORDERING)(x)
    f5 = x

    o = f4

    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(512, (3, 3), padding='valid' , activation='relu' , data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    o = attention_up_and_concate(o, f3, data_format=IMAGE_ORDERING)
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(256, (3, 3), padding='valid', activation='relu' , data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    o = attention_up_and_concate(o, f2, data_format=IMAGE_ORDERING)
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(128, (3, 3), padding='valid' , activation='relu' , data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    o = attention_up_and_concate(o, f1, data_format=IMAGE_ORDERING)
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(64, (3, 3), padding='valid', activation='relu', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    o = Conv2D(n_classes, (3, 3), padding='same',
                data_format=IMAGE_ORDERING)(o)
    o = ( UpSampling2D( (2,2), data_format='channels_last'))(o)
    o =  Conv2D(n_classes , (3, 3) , padding='same', data_format='channels_last' )( o )
    o = (Reshape((img_h*img_w, -1)))(o)

    o = (Activation('softmax'))(o)
    model = Model(img_input, o)

    #model.compile(optimizer=Adam(lr=1e-5), loss=[focal_loss()], metrics=['accuracy', dice_coef])
    return model

class AttentionVggUnet(VggUnet):

    def getVggUnetModel(self, layersNotTrain, inputImgSize, nClasses,loadWeights):
        model = att_unet(inputImgSize[0],inputImgSize[1],nClasses)
        #intermediate_layer_model = Model(inputs=model.input,outputs=model.layers[-4].output)
        for layer_num in range(layersNotTrain):
            model.layers[layer_num].trainable = False
        print ('Compiling model...') 
        model.compile(optimizer='adam', loss=loss_func, metrics=[dice_coef])
        if loadWeights is not None:
            model.load_weights(loadWeights)
        return model
