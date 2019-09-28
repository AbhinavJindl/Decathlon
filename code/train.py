import keras_segmentation
from keras.optimizers import Adam
from keras import backend as K

model = keras_segmentation.models.unet.vgg_unet(n_classes=3 ,  input_height=64, input_width=64)
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

# ...
model.compile(optimizer='adam', loss=dice_coef_loss, metrics=[dice_coef])
model.train( 
    train_images =  "train_images/input",
    train_annotations = "train_images/output",
    checkpoints_path = "vgg_unet_1" , epochs=5,batch_size=4
)

out = model.predict_segmentation(
    inp="train_images/input/hippocampus_001.nii.gz0_3.png",
    out_fname="out.png"
)




import matplotlib.pyplot as plt
plt.imshow(out)
