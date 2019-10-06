import nibabel as nib 
import os
import numpy as np
import matplotlib.image as Image


def print_dimensions(task_dir):
    train_image_list = os.listdir(task_dir + '/imagesTr')
    for train_image_name in train_image_list:
        if(train_image_name[0]!='.'):
            train_img_x = nib.load(task_dir + '/imagesTr/' + train_image_name).get_data()
            train_img_x = np.asarray(train_img_x)
            train_img_y = nib.load(task_dir + '/labelsTr/' + train_image_name).get_data()
            train_img_y = np.asarray(train_img_y)
            print (train_image_name,train_img_x.shape, train_img_y.shape)
            


if __name__=="__main__":
    DATASET_DIR = "../Dataset/"
    task_dirs = os.listdir(DATASET_DIR)
    for task_dir in task_dirs:
        print_dimensions(DATASET_DIR + task_dir)                