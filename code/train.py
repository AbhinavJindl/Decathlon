import keras_segmentation
from keras.optimizers import Adam
from keras import backend as K
import nibabel as nib
import numpy as np
from preproc import preprocess_image
from sklearn.model_selection import KFold
import os
import tensorflow as tf


def train_generator(img_path_list, label_path_list, batch_size, img_size):
    while True:
        for j in range(0, len(img_path_list), batch_size):
            train_x = []
            train_y = []
            for i in range(batch_size):
                if(i+j >= len(img_path_list)):
                    break
                img_data = nib.load(img_path_list[i+j]).get_fdata()
                label_data = nib.load(label_path_list[i+j]).get_fdata()
                tr_x,tr_y = preprocess_image(img_data,label_data,img_size)
                train_x += tr_x
                train_y += tr_y
                print('sample number ' + str(i+j))
            #print(np.array(train_x).shape)
            train_x = np.array(train_x)
            train_y = np.array(train_y)

            train_x = np.expand_dims(train_x, axis=3)
            train_y = np.expand_dims(train_y,axis=3)

            train_x = np.tile(train_x,(1,1,1,3))
            train_y = np.tile(train_y,(1,1,1,3))
            
            yield( train_x, train_y)


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

    checkpoint_path = "saved_models/training_heart/cp-{epoch:04d}.ckpt"
    # checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
                    filepath=checkpoint_path, 
                    verbose=1, 
                    save_weights_only=True,
                    period=5)



    img_size = (224,224)
    k_fold_train_data,k_fold_test_data = create_cross_validation_data(['../Dataset/Task02_Heart'])
    
    
    train_img_path_list = [i[0] for i in k_fold_train_data[0]]
    train_labels_path_list = [i[1] for i in k_fold_train_data[0]]
    
    test_img_path_list = [i[0] for i in k_fold_test_data[0]]
    test_labels_path_list = [i[1] for i in k_fold_test_data[0]]
    batch_size = 1

    model = keras_segmentation.models.unet.vgg_unet(n_classes=3 ,  input_height=img_size[0], input_width=img_size[1])
    print(model.summary())
    model.compile(optimizer='adam', loss=dice_coef_loss, metrics=[dice_coef])
    model.fit_generator(generator=train_generator(train_img_path_list,train_labels_path_list,batch_size=batch_size, img_size = img_size),
                        steps_per_epoch=len(train_img_path_list)/batch_size, 
                        epochs=5,
                        callbacks=[cp_callback], 
                        verbose=1,
                        validation_data = train_generator(test_img_path_list,test_labels_path_list,batch_size=batch_size, img_size = img_size),
                        validation_steps=len(test_img_path_list)/batch_size)



