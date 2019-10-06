import keras_segmentation
from keras.optimizers import Adam
from keras import backend as K
import nibabel as nib
import numpy as np
from preproc import preprocess_image,save_slices
from sklearn.model_selection import KFold
import os
import tensorflow as tf
import gc
import shutil


def train_generator(img_path_list, label_path_list, img_size, labelimg_size, per_img_batch_size=16,nClasses = 3):
    j = 0
    while True:
        gc.collect()
        train_x = []
        train_y = []
        img_data = nib.load(img_path_list[j]).get_fdata()
        label_data = nib.load(label_path_list[j]).get_fdata()
        tr_x,tr_y = preprocess_image(img_data,label_data,img_size,labelimg_size, nClasses)
        train_x += tr_x
        train_y += tr_y
        #print('sample number is ' + str(j))
        j+=1
        #print(np.array(train_x).shape)
        train_x = np.array(train_x)
        train_y = np.array(train_y)

        train_x = np.expand_dims(train_x, axis=3)
        #train_y = np.expand_dims(train_y,axis=3)

        train_x = np.tile(train_x,(1,1,1,3))
        #train_y = np.tile(train_y,(1,1,1,3))
        #print('sample shape is ' + str(train_x.shape))
        for k in range(0,train_x.shape[0], per_img_batch_size):
            end = min(k+per_img_batch_size, train_x.shape[0])
            yield( train_x[k:end], train_y[k:end])


def find_num_steps_train(train_img_list):
    tot_steps = 0
    for train_img in train_img_list:
        data = nib.load(train_img).get_data()
        for dim in data.shape:
            tot_steps += dim
    return tot_steps

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
    
    #First train individual organs.
    #-----------------
    #update this list to train individual organs.
    #-----------------
    task_dir_list = ['Task04_Hippocampus']
    for task_dir in task_dir_list:
        k_fold_train_data,k_fold_test_data = create_cross_validation_data(['../Dataset/' + task_dir])
        
        for fold in range(len(k_fold_train_data)):
            train_img_path_list = [i[0] for i in k_fold_train_data[fold]]
            train_labels_path_list = [i[1] for i in k_fold_train_data[fold]]
            
            test_img_path_list = [i[0] for i in k_fold_test_data[fold]]
            test_labels_path_list = [i[1] for i in k_fold_test_data[fold]]

            if os.path.exists('../saved_models') == False:
                os.mkdir('../saved_models')

            if os.path.exists('../saved_models/' + task_dir) == False:
                os.mkdir('../saved_models/' + task_dir)
            if (os.path.exists("../saved_models/"+task_dir+"/" +str(fold))) == False:
                os.mkdir("../saved_models/"+task_dir+"/" +str(fold))

            checkpoint_path = "../saved_models/"+task_dir+"/" +str(fold)+"/cp-{epoch:04d}.ckpt"
            cp_callback = tf.keras.callbacks.ModelCheckpoint(
                    filepath=checkpoint_path, 
                    verbose=1, 
                    save_weights_only=True,
                    save_freq='epoch')
            model = keras_segmentation.models.unet.vgg_unet(n_classes=3 ,  input_height=224, input_width=224)

            layers_not_cons = 14
            for layer_num in range(layers_not_cons):
                model.layers[layer_num].trainable = False

            model.compile(optimizer='adam', loss=dice_coef_loss, metrics=[dice_coef])
            per_image_batch_size = 16
            img_size = (model.input_height,model.input_width)
            labelimg_size = (model.output_height, model.output_width)
            steps_per_epoch = find_num_steps_train(train_img_path_list)
            model.fit_generator(generator=train_generator(train_img_path_list,train_labels_path_list, img_size = img_size,labelimg_size = labelimg_size, per_img_batch_size=per_image_batch_size),
                                steps_per_epoch=steps_per_epoch/per_image_batch_size, 
                                epochs=5,
                                callbacks=[cp_callback], 
                                verbose=1)



