import nibabel as nib 
import os
import numpy as np
import matplotlib.image as Image
import cv2

def saveImage(train_img_x, train_img_y, train_image_name, train_dir, axis, i):
    #print(train_img_y.dtype)
    # png.from_array(np.squeeze(train_img_x), 'L').save(train_dir + '/input/' + train_image_name + str(axis) + '_' +  str(i) + '.jpg')
    # png.from_array(np.squeeze(train_img_y), 'L').save(train_dir + '/ouptut/' + train_image_name + str(axis) + '_' +  str(i) + '.jpg')
    
    cv2.imwrite(train_dir + '/input/' + train_image_name + str(axis) + '_' +  str(i) + '.png', np.squeeze(train_img_x))
    cv2.imwrite(train_dir + '/output/' + train_image_name + str(axis) + '_' +  str(i)+ '.png', np.squeeze(train_img_y).astype(int))


def preprocess_image(img, labelimg, img_size):
    if(img.ndim == 3):
        img = np.reshape(img,(img.shape[0],img.shape[1],img.shape[2],1))
    #labelimg = np.reshape(labelimg,(img.shape[0],img.shape[1],img.shape[2],img.shape[3]))
    train_x = []
    train_y = []
    for j in range(img.shape[3]):
        for i in range(img.shape[0]):
            train_x.append(cv2.resize(img[i,:,:,j],img_size))
            train_y.append(cv2.resize(labelimg[i,:,:],img_size))
            
        for i in range(img.shape[1]):
            train_x.append(cv2.resize(img[:,i,:,j],img_size))
            train_y.append(cv2.resize(labelimg[:,i,:],img_size))
            
        for i in range(img.shape[2]):
            train_x.append(cv2.resize(img[:,:,i,j],img_size))
            train_y.append(cv2.resize(labelimg[:,:,i],img_size))
    return train_x,train_y

                


                    

