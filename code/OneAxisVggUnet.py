from vggUnet import VggUnet
from HelperFunctions import getSliceFromImage
import nibabel as nib
import numpy as np
from random import sample,shuffle


class OneAxisVggUnet(VggUnet):
    def preprocImage(self, img, labelimg):
        if(img.ndim == 3):
            img = np.reshape(img,(img.shape[0],img.shape[1],img.shape[2],1))

        pos_list = []
        neg_list = []
        #axis = np.argmin(np.array(img.shape[:-1]))
        axis = 2
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