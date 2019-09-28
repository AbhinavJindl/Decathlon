import nibabel as nib 
import os
import numpy as np
import matplotlib.image as Image
import png 
import cv2

def saveImage(train_img_x, train_img_y, train_image_name, train_dir, axis, i):
    #print(train_img_y.dtype)
    # png.from_array(np.squeeze(train_img_x), 'L').save(train_dir + '/input/' + train_image_name + str(axis) + '_' +  str(i) + '.jpg')
    # png.from_array(np.squeeze(train_img_y), 'L').save(train_dir + '/ouptut/' + train_image_name + str(axis) + '_' +  str(i) + '.jpg')
    
    cv2.imwrite(train_dir + '/input/' + train_image_name + str(axis) + '_' +  str(i) + '.png', np.squeeze(train_img_x))
    cv2.imwrite(train_dir + '/output/' + train_image_name + str(axis) + '_' +  str(i)+ '.png', np.squeeze(train_img_y).astype(int))

if __name__ == '__main__':
    #dirs = os.listdir(os.getcwd())
    
    task_dirs = ['Task04_Hippocampus']
    train_dir = 'train_images'
    for task_dir in task_dirs:
        train_image_list = os.listdir(os.path.join(task_dir + '/imagesTr'))
        for train_image_name in train_image_list:
            print(train_image_name)
            if(train_image_name[0]!='.'):
                train_img_x = nib.load(task_dir + '/imagesTr/' + train_image_name).get_data()
                train_img_x = np.asarray(train_img_x)
                train_img_y = nib.load(task_dir + '/labelsTr/' + train_image_name).get_data()
                train_img_y = np.asarray(train_img_y)
                for i in range(train_img_x.shape[0]):
                    saveImage(train_img_x[i,:,:],train_img_y[i,:,:],train_image_name,train_dir, 0,i)
                    
                for i in range(train_img_x.shape[1]):
                    saveImage(train_img_x[:,i,:],train_img_y[:,i,:], train_image_name,train_dir,1,i)
                    
                for i in range(train_img_x.shape[2]):
                    saveImage(train_img_x[:,:,i],train_img_y[:,:,i], train_image_name,train_dir,2,i)
                    
                #img = Image.imread(train_dir + '/output/' + train_image_name + str(0) + '_' +  str(0) + '.png')
                


                    

