import nibabel as nib 
import os
import numpy as np
import matplotlib.image as Image
import cv2

def saveImage(train_img_x, train_img_y, train_image_name, train_dir):
    #print(train_img_y.dtype)
    # png.from_array(np.squeeze(train_img_x), 'L').save(train_dir + '/input/' + train_image_name + str(axis) + '_' +  str(i) + '.jpg')
    # png.from_array(np.squeeze(train_img_y), 'L').save(train_dir + '/ouptut/' + train_image_name + str(axis) + '_' +  str(i) + '.jpg')
    
    cv2.imwrite(train_dir + '/input/' + train_image_name +'.png', np.squeeze(train_img_x))
    cv2.imwrite(train_dir + '/output/' + train_image_name + '.png', np.squeeze(train_img_y).astype(int))

def get_img_arr(img, img_size):
    img = cv2.resize(img, img_size)
    img = img.astype(np.float32)
    img = img/255.0
    return img

def get_seg_arr(img, img_size, nClasses):
    seg_labels = np.zeros((  img_size[0] , img_size[1]  , nClasses ))
    img = cv2.resize(img, img_size , interpolation=cv2.INTER_NEAREST )
    #img = img[:, : , 0]

    for c in range(nClasses):
        seg_labels[: , : , c ] = (img == c ).astype(int)

    seg_labels = np.reshape(seg_labels, ( img_size[0]*img_size[1] , nClasses ))
    return seg_labels

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

    kf = KFold(n_splits=folds)
    kf.get_n_splits(data_list)


    train_data = []
    test_data = []
    for train_index, test_index in kf.split(data_list):
        train_data.append([data_list[x] for x in train_index])
        test_data.append([data_list[x] for x in test_index])

    
    return train_data,test_data



def preprocess_image(img, labelimg, img_size, labelimg_size, nClasses):
    if(img.ndim == 3):
        img = np.reshape(img,(img.shape[0],img.shape[1],img.shape[2],1))
    #labelimg = np.reshape(labelimg,(img.shape[0],img.shape[1],img.shape[2],img.shape[3]))
    train_x = []
    train_y = []
    for j in range(img.shape[3]):
        for i in range(img.shape[0]):
            train_x.append(get_img_arr(img[i,:,:,j], img_size))
            train_y.append(get_seg_arr(labelimg[i,:,:],labelimg_size,nClasses))
            
        for i in range(img.shape[1]):
            train_x.append(get_img_arr(img[:,i,:,j], img_size))
            train_y.append(get_seg_arr(labelimg[:,i,:],labelimg_size,nClasses))
            
            
        for i in range(img.shape[2]):
            train_x.append(get_img_arr(img[:,:,i,j], img_size))
            train_y.append(get_seg_arr(labelimg[:,:,i],labelimg_size,nClasses))
           
    return train_x,train_y

def save_slices(image_list, label_list, save_dir):

    for k in range(len(image_list)):
        img_data = nib.load(image_list[k]).get_data()
        lab_data = nib.load(label_list[k]).get_data()
        if(img_data.ndim < 4):
            img_data = np.reshape(img_data, (img_data.shape[0],img_data.shape[1],img_data.shape[2],1))
        
        for j in range(img_data.shape[3]):
            for i in range(img_data.shape[0]):
                img_name = os.path.basename(image_list[k]) + '_' + str(j) + '_0' + '_' + str(i)
                saveImage(img_data[i,:,:,j], lab_data[i,:,:], img_name, save_dir) 
            for i in range(img_data.shape[1]):
                img_name = os.path.basename(image_list[k]) + '_' + str(j) + '_1' + '_' + str(i)
                saveImage(img_data[:,i,:,j], lab_data[:,i,:], img_name, save_dir)
                
            for i in range(img_data.shape[2]):
                img_name = os.path.basename(image_list[k]) + '_' + str(j) + '_2' + '_' + str(i)
                saveImage(img_data[:,:,i,j], lab_data[:,:,i], img_name, save_dir)
        #return

        




                


                    

