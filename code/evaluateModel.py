import nibabel as nib
import cv2
from HelperFunctions import load_pickle,getModalityType
from vggUnet import VggUnet
from OneAxisVggUnet import OneAxisVggUnet
from ThreeConsSliceVggUnet import ThreeConsSliceVggUnet
from OneAxisThreeConsSliceVggUnet import OneAxisThreeConsSliceVggUnet
from preprocessing import CtImagePreprocessor, NonCtImagePreprocessor
from trainVggUnetModel import loadVggUnet
import numpy as np
import os
from trainVggUnetModel import loadVggUnet

# def loadVggUnet(foldsDataFileName, taskDir, nClasses, imgSize, labelImgSize, loadWeights, folds, modalityType, modelType):
#   if not os.path.exists(foldsDataFileName):
#       print ('Creating folds....')
#       kFoldTrainData,kFoldTestData = create_cross_validation_data(['../Dataset/'+taskDir], folds)
#       print ('Saving folds...')
#       save_pickle((kFoldTrainData,kFoldTestData),foldDataFile)
#   (kFoldTrainData,kFoldTestData) = load_pickle(foldsDataFileName)
#   fold = 0
#   trainImgPathList = [i[0] for i in kFoldTrainData[fold]]
#   trainLabelPathList = [i[1] for i in kFoldTrainData[fold]]
#   preproc = None
#   if modalityType == 'CT':
#       preproc = CtImagePreprocessor(trainImgPathList,trainLabelPathList,imgSize,labelImgSize,nClasses)
#   else:
#       preproc = NonCtImagePreprocessor(trainImgPathList,trainLabelPathList,imgSize,labelImgSize,nClasses)
#   if modelType == 'VggUnet':
#       model = VggUnet(layersNotTrain=0,
#                           nClasses = nClasses,
#                           inputImgSize = imgSize,
#                           preproc=preproc,
#                           loadWeights=loadWeights)
#   elif modelType == 'OneAxisVggUnet':
#       model = OneAxisVggUnet(layersNotTrain=0,
#                           nClasses = nClasses,
#                           inputImgSize = imgSize,
#                           preproc=preproc,
#                           loadWeights=loadWeights)
#   elif modelType == 'ThreeConsSliceVggUnet':
#       model = ThreeConsSliceVggUnet(layersNotTrain=0,
#                           nClasses = nClasses,
#                           inputImgSize = imgSize,
#                           preproc=preproc,
#                           loadWeights=loadWeights)
#   elif modelType == 'OneAxisThreeConsSliceVggUnet':
#       model = OneAxisThreeConsSliceVggUnet(layersNotTrain=0,
#                           nClasses = nClasses,
#                           inputImgSize = imgSize,
#                           preproc=preproc,
#                           loadWeights=loadWeights)
#   elif modelType == 'AutoEncoderDecoderModel':
#       model = AutoEncoderDecoderModel(inputImgSize = imgSize,
#                           preproc=preproc,
#                           loadWeights=loadWeights)
#   return model



def getLabelImgClassWise(labelData, nClasses):
    labelImgClassWise = np.zeros(labelData.shape + (nClasses,))
    for c in range(nClasses):
        labelImgClassWise[: , : ,: , c ] = (labelData == c ).astype(int)
    return labelImgClassWise

def diceScore(actData, predData, nClasses):
    actImgClassWise = getLabelImgClassWise(actData, nClasses)
    predImgClassWise = getLabelImgClassWise(predData, nClasses)
    diceScoreList = []
    for i in range(1,nClasses):
        num = 2*np.sum(np.multiply(actImgClassWise[:,:,:,i], predImgClassWise[:,:,:,i]))
        den = np.sum(actImgClassWise[:,:,:,i]) + np.sum(predImgClassWise[:,:,:,i])
        diceScoreList.append(num/float(den))
    return diceScoreList


def evaluateModel(model, foldsDataFileName):
    (kFoldTrainData,kFoldTestData) = load_pickle(foldsDataFileName)
    fold = 0
    testImgPathList = [i[0] for i in kFoldTestData[fold]]
    testLabelPathList = [i[1] for i in kFoldTestData[fold]]
    nClasses = model.nClasses
    diceScoreList = []
    for i in range(len(testImgPathList)):
        imgData = nib.load(testImgPathList[i]).get_fdata()
        actData = nib.load(testLabelPathList[i]).get_fdata()
        predData = model.predict(imgData)
        diceScorei = diceScore(actData,predData,nClasses)
        print(testImgPathList[i], diceScorei)
        diceScoreList.append(diceScorei)
    diceScoreList = np.array(diceScoreList)
    meanValue = np.mean(diceScoreList, axis = 0)
    stdValue = np.std(diceScoreList, axis = 0)
    return meanValue, stdValue

if __name__ == '__main__':
    taskDir = 'Task04_Hippocampus'
    foldsDataFileName = 'folds_data_'+taskDir+'.pkl'
    nClasses = 2
    epochsN = 20
    batchSize = 32
    folds = 1
    saveDir = os.path.join('../saved_models',taskDir,'AttentionVggUnet')
    imgSize = (224,224)
    labelImgSize = (224,224)
    modelType = 'EnsembleAttentionVggUnet'
    modalityType = getModalityType(taskDir)
    loadWeights= [os.path.join('../saved_models',taskDir,'EnsembleAttentionVggUnet','0','axis0','cp-0020.ckpt'),
                    os.path.join('../saved_models',taskDir,'EnsembleAttentionVggUnet','0','axis1','cp-0020.ckpt'),
                    os.path.join('../saved_models',taskDir,'EnsembleAttentionVggUnet','0','axis2','cp-0020.ckpt')]
    layersNotTrain = 0
    print("time is 16:34")
    model = loadVggUnet(foldsDataFileName, taskDir, nClasses, imgSize, labelImgSize, loadWeights, folds, modalityType, modelType, layersNotTrain)
    print(evaluateModel(model, foldsDataFileName))