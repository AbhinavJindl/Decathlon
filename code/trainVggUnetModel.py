import os
from preprocessing import create_cross_validation_data
from preprocessing import NonCtImagePreprocessor,CtImagePreprocessor
from vggUnet import VggUnet
from OneAxisVggUnet import OneAxisVggUnet
from ThreeConsSliceVggUnet import ThreeConsSliceVggUnet
from OneAxisThreeConsSliceVggUnet import OneAxisThreeConsSliceVggUnet
from AutoEncoderDecoderModel import AutoEncoderDecoderModel
from EnsembleAttentionVggUnet import EnsembleAttentionVggUnet
from AttentionVggUnet import AttentionVggUnet
from HelperFunctions import save_pickle,load_pickle,getModalityType

def loadVggUnet(foldsDataFileName, taskDir, nClasses, imgSize, labelImgSize, loadWeights, folds, modalityType, modelType, layersNotTrain):
    if not os.path.exists(foldsDataFileName):
        print ('Creating folds....')
        kFoldTrainData,kFoldTestData = create_cross_validation_data(['../Dataset/'+taskDir], folds)
        print ('Saving folds...')
        save_pickle((kFoldTrainData,kFoldTestData),foldDataFile)
    (kFoldTrainData,kFoldTestData) = load_pickle(foldsDataFileName)
    fold = 0
    trainImgPathList = [i[0] for i in kFoldTrainData[fold]]
    trainLabelPathList = [i[1] for i in kFoldTrainData[fold]]
    preproc = None
    if modalityType == 'CT':
        preproc = CtImagePreprocessor(trainImgPathList,trainLabelPathList,imgSize,labelImgSize,nClasses)
    else:
        preproc = NonCtImagePreprocessor(trainImgPathList,trainLabelPathList,imgSize,labelImgSize,nClasses)
    if modelType == 'VggUnet':
        model = VggUnet(layersNotTrain=layersNotTrain,
                            nClasses = nClasses,
                            inputImgSize = imgSize,
                            preproc=preproc,
                            loadWeights=loadWeights)
    elif modelType == 'OneAxisVggUnet':
        model = OneAxisVggUnet(layersNotTrain=layersNotTrain,
                            nClasses = nClasses,
                            inputImgSize = imgSize,
                            preproc=preproc,
                            loadWeights=loadWeights)
    elif modelType == 'ThreeConsSliceVggUnet':
        model = ThreeConsSliceVggUnet(layersNotTrain=layersNotTrain,
                            nClasses = nClasses,
                            inputImgSize = imgSize,
                            preproc=preproc,
                            loadWeights=loadWeights)
    elif modelType == 'OneAxisThreeConsSliceVggUnet':
        model = OneAxisThreeConsSliceVggUnet(layersNotTrain=layersNotTrain,
                            nClasses = nClasses,
                            inputImgSize = imgSize,
                            preproc=preproc,
                            loadWeights=loadWeights)
    elif modelType == 'OneAxisThreeConsSliceVggUnetSelfSupervised':
        model = OneAxisThreeConsSliceVggUnet(layersNotTrain=layersNotTrain,
                            nClasses = nClasses,
                            inputImgSize = imgSize,
                            preproc=preproc,
                            loadWeights=None)
        model2 = AutoEncoderDecoderModel(inputImgSize = imgSize,
            preproc = preproc,
            loadWeights = loadWeights)
        for i in range(layersNotTrain):
            #print('loading layer number ',i)
            weights = model2.model.layers[i].get_weights()
            model.model.layers[i].set_weights(weights)
        #print('model loaded.')
    elif modelType == 'AutoEncoderDecoderModel':
        model = AutoEncoderDecoderModel(inputImgSize = imgSize,
                            preproc=preproc,
                            loadWeights=loadWeights)
    elif modelType == 'AttentionVggUnet':
        model = AttentionVggUnet(layersNotTrain=layersNotTrain,
                            nClasses = nClasses,
                            inputImgSize = imgSize,
                            preproc=preproc,
                            loadWeights=loadWeights)
    elif modelType == 'EnsembleAttentionVggUnet':
        #In this case give loadweights as a list.
        model = EnsembleAttentionVggUnet(layersNotTrain=layersNotTrain,
                            nClasses = nClasses,
                            inputImgSize = imgSize,
                            preproc=preproc,
                            loadWeights=loadWeights)
    return model



def trainVggUnetModel(taskDir, foldsDataFileName, nClasses, epochsN, batchSize, folds, saveDir, imgSize, labelImgSize, modelType, modalityType, loadWeights, layersNotTrain):
    model = loadVggUnet(foldsDataFileName, taskDir, nClasses, imgSize, labelImgSize, loadWeights, folds, modalityType, modelType, layersNotTrain)
    print('Training Started.....')
    (kFoldTrainData,kFoldTestData) = load_pickle(foldsDataFileName)
    fold = 0
    trainImgPathList = [i[0] for i in kFoldTrainData[fold]]
    trainLabelPathList = [i[1] for i in kFoldTrainData[fold]]
    savedModelDir = os.path.join(saveDir, str(fold))
    model.train(trainImgPathList, trainLabelPathList, savedModelDir, epochsN, batchSize, loadWeights=None)




if __name__ == '__main__':
    taskDir = 'Task10_Colon'
    foldsDataFileName = 'folds_data_'+taskDir+'.pkl'
    nClasses = 2
    epochsN = 20
    batchSize = 32
    folds = 1
    saveDir = os.path.join('../saved_models',taskDir,'EnsembleAttentionVggUnet')
    imgSize = (224,224)
    labelImgSize = (224,224)
    modelType = 'EnsembleAttentionVggUnet'
    modalityType = getModalityType(taskDir)
    loadWeights= [None,None,None]
    layersNotTrain = 0
    print('updated time 12:37')
    print (taskDir)
    trainVggUnetModel(taskDir, foldsDataFileName, nClasses,epochsN,batchSize,folds,saveDir,imgSize,labelImgSize,modelType,modalityType,loadWeights,layersNotTrain)