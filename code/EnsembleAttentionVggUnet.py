from AttentionVggUnet import AttentionVggUnet
from HelperFunctions import getSliceFromImage
import nibabel as nib
import numpy as np
from random import sample,shuffle
import os
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

class EnsembleAttentionVggUnet():
    def __init__(self, layersNotTrain, nClasses, inputImgSize, preproc, loadWeights):
        self.model1 = AttentionVggUnet(layersNotTrain=layersNotTrain,nClasses = nClasses,inputImgSize = inputImgSize,preproc=preproc,loadWeights=loadWeights[0]).model
        self.model2 = AttentionVggUnet(layersNotTrain=layersNotTrain,nClasses = nClasses,inputImgSize = inputImgSize,preproc=preproc,loadWeights=loadWeights[1]).model
        self.model3 = AttentionVggUnet(layersNotTrain=layersNotTrain,nClasses = nClasses,inputImgSize = inputImgSize,preproc=preproc,loadWeights=loadWeights[2]).model
        self.preproc = preproc
        self.nClasses = nClasses
        self.inputImgSize = inputImgSize

    def trainGenerator(self, trainImgPathList, trainLabelPathList, batchSize, axis):
        j = 0
        while True:
            imgData = nib.load(trainImgPathList[j]).get_fdata()
            labelData = nib.load(trainLabelPathList[j]).get_fdata()
            #This is used for Cropping
            #(minLeftColVal,maxRightColVal,minUpperRowVal,maxDownRowVal) = self.preproc.cropZeroValues(imgData, pixThresh=5,countThresh=0.4)
            trainX, trainY = self.preprocImage(imgData,labelData,axis)

            j = (j+1)%len(trainImgPathList) 
            for k in range(0,trainX.shape[0], batchSize):
                end = min(k+batchSize, trainX.shape[0])
                yield( trainX[k:end], trainY[k:end])

    def preprocImage(self, img, labelimg, axis):
        if(img.ndim == 3):
            img = np.reshape(img,(img.shape[0],img.shape[1],img.shape[2],1))

        pos_list = []
        neg_list = []
        for j in range(img.shape[3]):
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

    def findNumberOfSteps(self, trainImgPathList, batchSize, axis):
        totSteps = 0
        for trainImgPath in trainImgPathList:
            h = nib.load(trainImgPath).header
            totSteps += h.get_data_shape()[axis]
        return totSteps/batchSize

    def trainModelAlongGivenAxis(self, trainImgPathList, trainLabelPathList, model, axis, savedModelDir, epochsN, batchSize, loadWeights = None):
        os.makedirs(savedModelDir,exist_ok=True)
        checkpoint_path = os.path.join(savedModelDir,"cp-{epoch:04d}.ckpt")
        cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, 
                                                      verbose=1, 
                                                      save_weights_only=True,
                                                      period=1)

        csv_logger = CSVLogger(os.path.join(savedModelDir,'model_save_history.csv'), append=True)
        if loadWeights is not None:
          self.model.load_weights(loadWeights)
        steps_per_epoch = self.findNumberOfSteps(trainImgPathList,batchSize,axis)
        his = model.fit_generator(generator=self.trainGenerator(trainImgPathList,trainLabelPathList, batchSize,axis),
                                      steps_per_epoch=steps_per_epoch,
                                      epochs=epochsN,
                                      callbacks=[cp_callback,csv_logger], 
                                      verbose=1)

    def train(self, trainImgPathList,trainLabelPathList, savedModelDir, epochsN, batchSize, loadWeights = None):
        #self.trainModelAlongGivenAxis(trainImgPathList, trainLabelPathList,self.model1, 0, os.path.join(savedModelDir,'axis0'), epochsN, batchSize)
        #self.trainModelAlongGivenAxis(trainImgPathList, trainLabelPathList,self.model2, 1, os.path.join(savedModelDir,'axis1'), epochsN, batchSize)
        self.trainModelAlongGivenAxis(trainImgPathList, trainLabelPathList,self.model3, 2, os.path.join(savedModelDir,'axis2'), epochsN, batchSize)
        


    def predict(self, img):
        predImg1 = self.predictUtil(img, self.model1, 0)
        predImg2 = self.predictUtil(img, self.model2, 1)
        predImg3 = self.predictUtil(img, self.model3, 2)
        predImg = predImg1# + predImg2 + predImg3
        predImg = np.where(predImg > 0, 1, 0)
        return predImg

    def predictUtil(self, img, model, axis):
        if(img.ndim == 3):
            img = np.reshape(img, img.shape + (1,))
        #(minLeftColVal,maxRightColVal,minUpperRowVal,maxDownRowVal) = self.preproc.getCroppingParameters()
        predImg = np.zeros(img.shape[:-1])
        minAxis = axis
        #img = img[minUpperRowVal:maxDownRowVal,minLeftColVal:maxRightColVal]
        for j in range(img.shape[3]):
            for i in range(img.shape[minAxis]):
                actImgSlice = getSliceFromImage(img, minAxis, i, j)
                imgSlice = self.preproc.getPreprocImgSlice(actImgSlice)
                imgSlice = np.expand_dims(imgSlice, axis=2)
                imgSlice = np.tile(imgSlice,(1,1,1,3))
                predSlice = model.predict(imgSlice)[0]
                predSlice = predSlice.reshape((self.inputImgSize[1], self.inputImgSize[0], self.nClasses)).argmax( axis=2 )
                # predSlice = cv2.resize(predSlice, (maxRightColVal-minLeftColVal , maxDownRowVal-minUpperRowVal) , interpolation=cv2.INTER_NEAREST)
                # predImg[minUpperRowVal:maxDownRowVal, minLeftColVal:maxRightColVal, i] = predSlice
                predSlice = cv2.resize(predSlice, (actImgSlice.shape[1],actImgSlice.shape[0]), interpolation=cv2.INTER_NEAREST)
                if (minAxis == 0):
                    predImg[i,:,:] = predSlice
                elif (minAxis == 1):
                    predImg[:,i,:] = predSlice
                else:
                    #predImg[minUpperRowVal:maxDownRowVal, minLeftColVal:maxRightColVal, i] = predSlice
                    predImg[:,:,i] = predSlice
                    

        return predImg

    def preprocImage(self, img, labelimg, axis):
        if(img.ndim == 3):
            img = np.reshape(img,(img.shape[0],img.shape[1],img.shape[2],1))

        pos_list = []
        neg_list = []
        #axis = np.argmin(np.array(img.shape[:-1]))
        for j in range(img.shape[3]):
            for i in range(1,img.shape[axis]-1):
                imgslice = self.preproc.getPreprocImgSlice(getSliceFromImage(img, axis, i, j))
                prevslice = self.preproc.getPreprocImgSlice(getSliceFromImage(img, axis, i-1, j))
                nextslice = self.preproc.getPreprocImgSlice(getSliceFromImage(img, axis, i+1, j))
                labelslice = getSliceFromImage(labelimg, axis, i, -1)
                preprocImgSlice = (prevslice, imgslice, nextslice)
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
        train_x = []
        for j in range(3):
            train_x.append([i[0][j] for i in samples_list])
        train_y = [i[1] for i in samples_list]
        tx = np.zeros(((len(train_x[0]),) + self.preproc.modelImgSize + (3,)))
        for i in range(3):
            tx[:,:,:,i] = np.array(train_x[i])
        return tx,np.array(train_y)