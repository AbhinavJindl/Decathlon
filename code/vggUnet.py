import keras_segmentation
from keras_segmentation.models.unet import vgg_unet
from keras_segmentation.models.model_utils import get_segmentation_model
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

def dice_coef(y_true, y_pred, smooth=0.01):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred,smooth=0.01):
    return 1-dice_coef(y_true, y_pred, smooth)

def loss_func(y_true,y_pred, smooth=0.01):
    return dice_coef_loss(y_true,y_pred,smooth) + keras.losses.categorical_crossentropy(y_true,y_pred)

def getLabelImgClassWise(labelData, nClasses):
    labelImgClassWise = np.zeros(labelData.shape + (nClasses,))
    for c in range(nClasses):
        labelImgClassWise[: , : ,: , c ] = (labelData == c ).astype(int)
    return labelImgClassWise


def getLargestCC(segmentation):
    labels = label(segmentation)
    if( labels.max() == 0 ):
      return labels == 0
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

def getlargestCCForAllLabels(labelData, nClasses):
    classWiseLabelData = getLabelImgClassWise(labelData, nClasses)
    for i in range(nClasses):
        c = getLargestCC(classWiseLabelData[:,:,:,i])
        classWiseLabelData[:,:,:,i] = np.where( c==True,classWiseLabelData[:,:,:,i],0)
    return classWiseLabelData.argmax(axis=3)


class VggUnet:
    def __init__(self, layersNotTrain, nClasses, inputImgSize, preproc, loadWeights=None):
        self.model = self.getVggUnetModel(layersNotTrain,inputImgSize,nClasses,loadWeights)
        self.preproc = preproc
        self.nClasses = nClasses
        self.inputImgSize = inputImgSize

    def preprocImage(self, img, labelimg):
        if(img.ndim == 3):
            img = np.reshape(img,(img.shape[0],img.shape[1],img.shape[2],1))

        pos_list = []
        neg_list = []
        for j in range(img.shape[3]):
            for axis in range(len(img.shape)-1):
                for i in range(img.shape[axis]):
                    imgslice = getSliceFromImage(img, axis, i, j)
                    labelslice = getSliceFromImage(labelimg, axis, i, -1)
                    preprocImgSlice = self.preproc.getPreprocImgSlice(imgslice)
                    preprocLabelSlice = self.preproc.getLabelWiseSlicedImage(labelslice)
                    if(len(labelslice[labelslice==1]) > 0):
                        pos_list.append([preprocImgSlice,preprocLabelSlice])
                    else:
                        neg_list.append([preprocImgSlice,preprocLabelSlice])
        samples_list = []
        if(len(pos_list) < len(neg_list)):
            neg_cons_list = sample(neg_list,k=len(pos_list))
            samples_list = neg_cons_list + pos_list
        else:
            pos_cons_list = sample(pos_list,k=len(neg_list))
            samples_list = neg_list + pos_cons_list
        shuffle(samples_list)
        train_x = [i[0] for i in samples_list]
        train_y = [i[1] for i in samples_list]
        tx = np.zeros(((len(train_x),) + self.preproc.modelImgSize + (3,)))
        for i in range(3):
            tx[:,:,:,i] = np.array(train_x)
        return tx,np.array(train_y)


    def getVggUnetModel(self, layersNotTrain, inputImgSize, nClasses,loadWeights):
        model = vgg_unet(n_classes=nClasses ,  input_height=inputImgSize[0], input_width=inputImgSize[1])
        #intermediate_layer_model = Model(inputs=model.input,outputs=model.layers[-4].output)
        o = model.layers[-4].output
        o = ( UpSampling2D( (2,2), data_format='channels_last'))(o)
        o =  Conv2D( model.n_classes , (3, 3) , padding='same', data_format='channels_last' )( o )
        model = get_segmentation_model(model.input, o)
        layers_not_cons = layersNotTrain
        for layer_num in range(layers_not_cons):
          model.layers[layer_num].trainable = False
        print ('Compiling model...') 
        model.compile(optimizer='adam', loss=loss_func, metrics=[dice_coef])
        if loadWeights is not None:
            model.load_weights(loadWeights)
        return model

    def trainGenerator(self, trainImgPathList, trainLabelPathList, batchSize):
        j = 0
        while True:
            imgData = nib.load(trainImgPathList[j]).get_fdata()
            labelData = nib.load(trainLabelPathList[j]).get_fdata()
            #This is used for Cropping
            # (minLeftColVal,maxRightColVal,minUpperRowVal,maxDownRowVal) = self.preproc.getCroppingParameters()
            # trainX, trainY = self.preprocImage(imgData[minUpperRowVal:maxDownRowVal, minLeftColVal:maxRightColVal], 
            #                                   labelData[minUpperRowVal:maxDownRowVal, minLeftColVal:maxRightColVal])
            trainX, trainY = self.preprocImage(imgData, labelData)
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

    def train(self, trainImgPathList,trainLabelPathList, savedModelDir, epochsN, batchSize, loadWeights = None):
        os.makedirs(savedModelDir,exist_ok=True)
        checkpoint_path = os.path.join(savedModelDir,"cp-{epoch:04d}.ckpt")
        cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, 
                                                        verbose=1, 
                                                        save_weights_only=True,
                                                        period=1)

        csv_logger = CSVLogger(os.path.join(savedModelDir,'model_save_history.csv'), append=True)
        if loadWeights is not None:
            self.model.load_weights(loadWeights)
        steps_per_epoch = self.findNumberOfSteps(trainImgPathList,batchSize)
        his = self.model.fit_generator(generator=self.trainGenerator(trainImgPathList,trainLabelPathList, batchSize),
                                        steps_per_epoch=steps_per_epoch,
                                        epochs=epochsN,
                                        callbacks=[cp_callback,csv_logger], 
                                        verbose=1)

    


    '''
        Assuming input to be 3dimensional

    '''
    def predict(self, img):
        if(img.ndim == 3):
            img = np.reshape(img, img.shape + (1,))
        #(minLeftColVal,maxRightColVal,minUpperRowVal,maxDownRowVal) = self.preproc.getCroppingParameters()
        predImg = np.zeros(img.shape[:-1])
        minAxis = 2
        #img = img[minUpperRowVal:maxDownRowVal,minLeftColVal:maxRightColVal]
        for j in range(img.shape[3]):
            for i in range(img.shape[minAxis]):
                actImgSlice = getSliceFromImage(img, minAxis, i, j)
                imgSlice = self.preproc.getPreprocImgSlice(actImgSlice)
                imgSlice = np.expand_dims(imgSlice, axis=2)
                imgSlice = np.tile(imgSlice,(1,1,1,3))
                predSlice = self.model.predict(imgSlice)[0]
                predSlice = predSlice.reshape((self.inputImgSize[1], self.inputImgSize[0], self.nClasses)).argmax( axis=2 )
                # predSlice = cv2.resize(predSlice, (maxRightColVal-minLeftColVal , maxDownRowVal-minUpperRowVal) , interpolation=cv2.INTER_NEAREST)
                # predImg[minUpperRowVal:maxDownRowVal, minLeftColVal:maxRightColVal, i] = predSlice
                predSlice = cv2.resize(predSlice, (actImgSlice.shape[1],actImgSlice.shape[0]), interpolation=cv2.INTER_NEAREST)
                predImg[:, :, i] = predSlice
                    

        return getlargestCCForAllLabels(predImg, self.nClasses)
    

