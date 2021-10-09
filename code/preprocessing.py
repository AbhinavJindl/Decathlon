from HelperFunctions import getSliceFromImage
import nibabel as nib
import numpy as np
from random import sample,shuffle
from sklearn.model_selection import KFold
import shutil
import cv2
import pickle
import os
import numpy as np
from sklearn.model_selection import KFold
from math import sqrt
from HelperFunctions import getSliceFromImage

class ImagePreprocessor:
    def __init__(self, imageFilesNameList, labelFilesNameList, modelImgSize, modelLabelImgSize,nClasses):
        self.imagesFilesNameList = imageFilesNameList
        self.labelFilesNameList = labelFilesNameList
        self.modelImgSize = modelImgSize
        self.modelLabelImgSize = modelLabelImgSize
        self.nClasses = nClasses
        #(self.minLeftColVal,self.maxRightColVal,self.minUpperRowVal,self.maxDownRowVal) = self.getCroppingParametersUtil(imageFilesNameList)


    def getCroppingParameters(self):
        return (self.minLeftColVal,self.maxRightColVal,self.minUpperRowVal,self.maxDownRowVal)

    def getLabelWiseSlicedImage(self, labelImgSlice):
        seg_labels = np.zeros(self.modelLabelImgSize + (self.nClasses,))
        img = cv2.resize(labelImgSlice, self.modelLabelImgSize , interpolation=cv2.INTER_NEAREST )
        for c in range(self.nClasses):
            seg_labels[: , : , c ] = (img == c ).astype(int)
        seg_labels = np.reshape(seg_labels, ( self.modelLabelImgSize[0]*self.modelLabelImgSize[1] , self.nClasses ))
        return seg_labels

    '''
        This function is used to get the preprocessed output of the given image Slice
    '''
    def getPreprocImgSlice(self,imgSlice):
        pass

    def cropZeroValues(self, img, pixThresh,countThresh):
        if(img.ndim == 4):
            img = img[:,:,:,0]

        minLeftColVal = img.shape[1]
        minUpperRowVal = img.shape[0]
        maxRightColVal = 0
        maxDownRowVal = 0
        for i in range(img.shape[2]):
            imgSlice = getSliceFromImage(img, 2, i, -1)
            imgSlice = cv2.normalize(imgSlice, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            imgSlice = np.where(imgSlice < pixThresh, 0, 1)
            zeroColumnCount = (imgSlice.shape[0] - np.count_nonzero(imgSlice, axis=0))/imgSlice.shape[0]
            
            leftColVal = 0
            for i in zeroColumnCount:
                if(i < countThresh):
                    break
                leftColVal += 1

            rightColVal = imgSlice.shape[1]-1
            for i in zeroColumnCount[::-1]:
                if(i < countThresh):
                    break
                rightColVal -= 1

            zeroRowCount = (imgSlice.shape[1] - np.count_nonzero(imgSlice, axis=1))/imgSlice.shape[1]

            upperRowVal = 0
            for i in zeroRowCount:
                if(i < countThresh):
                    break
                upperRowVal += 1

            downRowVal = imgSlice.shape[1]-1
            for i in zeroRowCount[::-1]:
                if(i < countThresh):
                    break
                downRowVal -= 1

            minLeftColVal = min(minLeftColVal,leftColVal)
            minUpperRowVal = min(minUpperRowVal,upperRowVal)
            maxRightColVal = max(maxRightColVal,rightColVal)
            maxDownRowVal = max(maxDownRowVal,downRowVal)
            
        return (minLeftColVal,maxRightColVal,minUpperRowVal,maxDownRowVal)  

    def getCroppingParametersUtil(self, imageFilesNameList ,pixThresh=5, countThresh=0.4):
        print('computing crop values for the dataset.')
        minLeftColVal = float('inf')
        minUpperRowVal = float('inf')
        maxRightColVal = 0
        maxDownRowVal = 0
        for imgFilesName in imageFilesNameList:
            imgData = nib.load(imgFilesName).get_fdata()
            (leftColVal,rightColVal,upperRowVal,downRowVal) = self.cropZeroValues(imgData, pixThresh, countThresh)
            minLeftColVal = min(minLeftColVal,leftColVal)
            minUpperRowVal = min(minUpperRowVal,upperRowVal)
            maxRightColVal = max(maxRightColVal,rightColVal)
            maxDownRowVal = max(maxDownRowVal,downRowVal)
        print('Cropping parameters for the dataset completed.')
        return (minLeftColVal,maxRightColVal,minUpperRowVal,maxDownRowVal)

    


class NonCtImagePreprocessor(ImagePreprocessor):

    def getPreprocImgSlice(self, imgSlice):
        imgSlice = cv2.resize(imgSlice, self.modelImgSize).astype(np.float32)
        imgSlice = cv2.normalize(imgSlice, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        return imgSlice

class CtImagePreprocessor(ImagePreprocessor):
    def __init__(self, imageFilesNameList, labelFilesNameList, modelImgSize, modelLabelImgSize,nClasses):
        ImagePreprocessor.__init__(self,imageFilesNameList, labelFilesNameList, modelImgSize, modelLabelImgSize,nClasses)
        # self.meanValue = self.calculateMeanImage()
        # self.stdValue = self.calculateStdImage()

    # def calculateMeanImage(self):
    #   meanValue = 0
    #   pixelCount = 0
    #   for imgName in self.imagesFilesNameList:
    #       imgData = nib.load(imgName).get_fdata()
    #       meanValue += np.sum(imgData)
    #       imgShape = imgData.shape
    #       imgPixelCount = 1
    #       for dim in imgShape:
    #           imgPixelCount *= dim
    #       pixelCount += imgPixelCount
    #   return meanValue/float(pixelCount)

    # def calculateStdImage(self):
    #   meanValue = self.meanValue
    #   pixelCount = 0
    #   stdValue = 0.0
    #   for imgName in self.imagesFilesNameList:
    #       imgData = nib.load(imgName).get_fdata()
    #       imgData = np.subtract(imgData,np.double(meanValue))
    #       imgData = np.multiply(imgData, imgData)
    #       stdValue = np.sum(imgData)
    #       imgShape = imgData.shape
    #       imgPixelCount = 1
    #       for dim in imgShape:
    #           imgPixelCount *= dim
    #       pixelCount += imgPixelCount
    #   return sqrt(stdValue/pixelCount)

    def getPreprocImgSlice(self,imgSlice):
        #imgSlice = (imgSlice - self.meanValue)/self.stdValue
        imgSlice = cv2.resize(imgSlice, self.modelImgSize).astype(np.float32)
        imgSlice = cv2.normalize(imgSlice, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        return imgSlice




def image_list_from_dir(task_dir):
    image_x_list = os.listdir(task_dir + '/imagesTr')
    image_y_list = os.listdir(task_dir + '/labelsTr')

    image_x_list = sorted(image_x_list)
    image_y_list = sorted(image_y_list)

    # storing images path after removing '.' beginning images
    image_x_list = [os.path.join(task_dir,'imagesTr',x) for x in image_x_list if x[0]!='.']
    image_y_list = [os.path.join(task_dir,'labelsTr',x) for x in image_y_list if x[0]!='.']

    return image_x_list,image_y_list

def create_cross_validation_data(task_dir_list, folds=5):

    # list of all images of all tasks
    image_x_list = []
    image_y_list = []
    for task_dir in task_dir_list:
        image_x, image_y = image_list_from_dir(task_dir)
        image_x_list += image_x
        image_y_list += image_y

    data_list = []
    for i in range(len(image_x_list)):
        data_list.append((image_x_list[i],image_y_list[i]))

    kf = KFold(n_splits=folds,shuffle=True)
    kf.get_n_splits(data_list)


    train_data = []
    test_data = []
    for train_index, test_index in kf.split(data_list):
        train_data.append([data_list[x] for x in train_index])
        test_data.append([data_list[x] for x in test_index])


    return train_data,test_data