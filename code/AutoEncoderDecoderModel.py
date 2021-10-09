import keras_segmentation
from keras import backend as K
from keras.models import Model
from keras.layers import *
from keras.callbacks import CSVLogger
import nibabel as nib
import numpy as np
from random import sample,shuffle
import shutil
import cv2
import keras
import tensorflow as tf
import os
from sklearn.model_selection import KFold
from preprocessing import create_cross_validation_data
from HelperFunctions import getSliceFromImage
from skimage.measure import label 
# class PredictionCallback(tf.keras.callbacks.Callback):    
#   def on_epoch_end(self, epoch, logs={}):
#       y_pred = self.model.predict(self.validation_data[0])
#       print('prediction: {} at epoch: {}'.format(y_pred, epoch))

class AutoEncoderDecoderModel():
    def __init__(self, inputImgSize, preproc, loadWeights=None):
        self.model = self.getVggUnetModel(inputImgSize,loadWeights)
        self.preproc = preproc

    def getVggUnetModel(self, inputImgSize, loadWeights):
        model = keras_segmentation.models.unet.vgg_unet(n_classes=1 ,  input_height=inputImgSize[0], input_width=inputImgSize[1])
        o = model.layers[-4].output
        o = ( UpSampling2D( (2,2), data_format='channels_last'))(o)
        o =  Conv2D( model.n_classes , (3, 3) , padding='same', data_format='channels_last' )( o )
        #model = keras_segmentation.models.model_utils.get_segmentation_model(model.input, o)
        o = (Reshape(( inputImgSize[0]*inputImgSize[1],-1)))(o)
        o = (Activation('relu'))(o)
        model = Model(model.layers[0].output, o)
        print ('Compiling model...') 
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        if loadWeights is not None:
            model.load_weights(loadWeights)
        print(model.summary())
        return model

    def preprocImage(self, img):
        if(img.ndim == 3):
            img = np.reshape(img,(img.shape[0],img.shape[1],img.shape[2],1))
        axis = 2
        imageList = []
        for j in range(img.shape[3]):
            for i in range(1,img.shape[axis]-1):
                imgslice = self.preproc.getPreprocImgSlice(getSliceFromImage(img, axis, i, j))
                prevslice = self.preproc.getPreprocImgSlice(getSliceFromImage(img, axis, i-1, j))
                nextslice = self.preproc.getPreprocImgSlice(getSliceFromImage(img, axis, i+1, j))
                preprocImgSlice = (prevslice, imgslice, nextslice)
                imageList.append([preprocImgSlice])
        shuffle(imageList)
        train_x = []
        for j in range(3):
            train_x.append([i[0][j] for i in imageList])
        tx = np.zeros(((len(train_x[0]),) + self.preproc.modelImgSize + (3,)))
        ty = [np.expand_dims(np.array(i[0][1]).flatten(),axis=1) for i in imageList]
        for i in range(3):
            tx[:,:,:,i] = np.array(train_x[i])
        return tx, np.array(ty)

    def trainGenerator(self, trainImgPathList, batchSize):
        j = 0
        while True:
            imgData = nib.load(trainImgPathList[j]).get_fdata()
            trainX, trainY = self.preprocImage(imgData)
            j = (j+1)%len(trainImgPathList) 
            for k in range(0,trainX.shape[0], batchSize):
                end = min(k+batchSize, trainX.shape[0])
                yield( trainX[k:end], trainY[k:end])

    def findNumberOfSteps(self, trainImgPathList, batchSize):
        totSteps = 0
        for trainImgPath in trainImgPathList:
            h = nib.load(trainImgPath).header
            totSteps += sum(h.get_data_shape())
        return totSteps/batchSize

    def train(self, trainImgPathList, trainLabelPathList, savedModelDir, epochsN, batchSize, loadWeights = None):
        os.makedirs(savedModelDir,exist_ok=True)
        checkpoint_path = os.path.join(savedModelDir,"cp-{epoch:04d}.ckpt")
        cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, 
                                                        verbose=1, 
                                                        save_weights_only=True,
                                                        period=1)
        csv_logger = CSVLogger(os.path.join(savedModelDir,'model_save_history.csv'), append=True)
        steps_per_epoch = self.findNumberOfSteps(trainImgPathList, batchSize)
        his = self.model.fit_generator(generator=self.trainGenerator(trainImgPathList, batchSize),
                                        steps_per_epoch=steps_per_epoch,
                                        epochs=epochsN,
                                        callbacks=[cp_callback,csv_logger], 
                                        verbose=1)

    def predict(self, img):
        if(img.ndim == 3):
            img = np.reshape(img, img.shape + (1,))
        predImg = np.zeros(img.shape[:-1])
        minAxis = 2
        img = img
        for j in range(img.shape[3]):
            for i in range(1,img.shape[minAxis]-1):
                imgslice = self.preproc.getPreprocImgSlice(getSliceFromImage(img, axis, i, j))
                prevslice = self.preproc.getPreprocImgSlice(getSliceFromImage(img, axis, i-1, j))
                nextslice = self.preproc.getPreprocImgSlice(getSliceFromImage(img, axis, i+1, j))
                finalslice = np.zeros(self.preproc.modelImgSize + (3,))
                finalslice[:,:,0] = prevslice
                finalslice[:,:,1] = imgslice
                finalslice[:,:,2] = nextslice
                predSlice = self.model.predict(finalslice)[0]
                predSlice = predSlice.reshape((imgslice.shape[0], imgslice.shape[1]))
                predSlice = cv2.resize(predSlice, (img.shape[1],img.shape[0]) , interpolation=cv2.INTER_CUBIC)
                predImg[:,:,i] = predSlice

        return predImg