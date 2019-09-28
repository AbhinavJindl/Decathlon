import keras_segmentation
from keras.optimizers import Adam
from keras import backend as K
import nibabel as nib
import numpy as np
from preproc import preprocess_image
from sklearn.model_selection import KFold
import os


def train_generator(img_path_list, label_path_list, batch_size, img_size):
    while True:
        for j in range(0, img_path_list, batch_size):
            train_x = []
            train_y = []
            for i in batch_size:
                img_data = nib.load(img_path_list[i+j]).get_fdata()
                label_data = nib.load(label_path_list[i+j]).get_fdata()
                tr_x,tr_y = preprocess_image(img_data,label_data,img_size)
                train_x += tr_x
                train_y += tr_y
            yield( np.array(train_x), np.array(train_y))


def image_list_from_dir(task_dir):
    image_x_list = os.listdir(task_dir + '/imagesTr')
    image_y_list = os.listdir(task_dir + '/labelsTr')


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

    kf = KFold(n_splits=folds)
    kf.get_n_splits(data_list)


    train_data = []
    test_data = []
    for train_index, test_index in kf.split(data_list):
        train_data.append([data_list[x] for x in train_index])
        test_data.append([data_list[x] for x in test_index])

    
    return train_data,test_data
                    



def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)



if __name__=="__main__":
    # model = keras_segmentation.models.unet.vgg_unet(n_classes=3 ,  input_height=64, input_width=64)
    # # ...
    # model.compile(optimizer='adam', loss=dice_coef_loss, metrics=[dice_coef])
    # model.train( 
    #     train_images =  "train_images/input",
    #     train_annotations = "train_images/output",
    #     checkpoints_path = "vgg_unet_1" , epochs=5,batch_size=4
    # )

    # out = model.predict_segmentation(
    #     inp="train_images/input/hippocampus_001.nii.gz0_3.png",
    #     out_fname="out.png"
    # )

    #print(create_cross_validation_data(['../Dataset/Task02_Heart']))
    model = keras_segmentation.models.unet.vgg_unet(n_classes=3 ,  input_height=64, input_width=64)


