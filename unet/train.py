from __future__ import print_function

import os
from skimage.io import imsave
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K

from data import load_train_data, load_test_data
from dataGenerator import *
from sklearn.model_selection import KFold

import SimpleITK as sitk

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

img_rows = 256
img_cols = 256

smooth = 0.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    # DICE = 2TP / (2TP+FP+FN)
    #tp = K.sum(y_true_f * y_pred_f)
    #fp = K.sum(K.maximum(y_pred_f-y_true_f,(y_true_f-y_true_f)))
    #fn = K.sum(K.maximum(y_true_f-y_pred_f,(y_true_f-y_true_f)))

    #return (2. * tp + smooth) / (2. * tp + fp + fn + smooth)

    # F1 Score
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_unet():
    inputs = Input((img_rows, img_cols, 1))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=1e-6), loss=dice_coef_loss, metrics=[dice_coef])

    return model


def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.float)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_cols, img_rows), preserve_range=True)

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p

'''def validatebatchGenerator():
    i = 0

    filetype = ['patient14', 'patient15', 'patient16', 'patient17','patient18', 'patient19', 'patient20', 'patient21']

    for filename in filetype:
        trainData = np.load(
            'D:\\Duong\\Data\\trainingsData\\unet\\' + filename + '_train.npy')  # 'imgs_train'+str(i)+'.npy')
        trainDataSeg = np.load(
            'D:\\Duong\\Data\\trainingsData\\unet\\' + filename + '_train_mask.npy')  # 'imgs_mask_train'+str(i)+'.npy')

        imgs_train = preprocess(trainData)
        imgs_mask_train = preprocess(trainDataSeg)

        # Convert to float32 for better precision
        imgs_train = imgs_train.astype('float32')
        imgs_mask_train = imgs_mask_train.astype('float32')

        mean = np.mean(imgs_train)  # mean for data centering
        std = np.std(imgs_train)  # std for data normalization

        imgs_train -= mean
        imgs_train /= std

        imgs_mask_train /= 2.  # scale masks to [0, 1]

        test = np.zeros((1, 256, 256, 1))
        test2 = np.zeros((1, 256, 256, 1))

        # Outputs image by image
        for imgTrain, imgMask in zip(imgs_train, imgs_mask_train):
            if i > 0:
                i = 0
                yield (test, test2)
                test = np.zeros((1, 256, 256, 1))
                test2 = np.zeros((1, 256, 256, 1))
            test[i] = imgTrain
            test2[i] = imgMask
            i += 1

def batchGenerator():
    i = 0

    filetype = ['patient01', 'patient02', 'patient03', 'patient04', 'patient05', 'patient06', 'patient07', 'patient08',
                'patient09', 'patient10', 'patient11', 'patient13', 'patient14', 'patient15', 'patient16', 'patient17',
                'patient18', 'patient19', 'patient20', 'patient21']

    for filename in filetype:
        trainData = np.load(
            'D:\\Duong\\Data\\trainingsData\\unet\\' + filename + '_train.npy')  # 'imgs_train'+str(i)+'.npy')
        trainDataSeg = np.load(
            'D:\\Duong\\Data\\trainingsData\\unet\\' + filename + '_train_mask.npy')  # 'imgs_mask_train'+str(i)+'.npy')

        imgs_train = preprocess(trainData)
        imgs_mask_train = preprocess(trainDataSeg)

        # Convert to float32 for better precision
        imgs_train = imgs_train.astype('float32')
        imgs_mask_train = imgs_mask_train.astype('float32')

        mean = np.mean(imgs_train)  # mean for data centering
        std = np.std(imgs_train)  # std for data normalization

        imgs_train -= mean
        imgs_train /= std

        imgs_mask_train /= 2.  # scale masks to [0, 1]

        test = np.zeros((1, 256, 256, 1))
        test2 = np.zeros((1, 256, 256, 1))

        # Outputs image by image
        for imgTrain, imgMask in zip(imgs_train, imgs_mask_train):
            if i > 0:
                i = 0
                yield (test, test2)
                test = np.zeros((1, 256, 256, 1))
                test2 = np.zeros((1, 256, 256, 1))
            test[i] = imgTrain
            test2[i] = imgMask
            i += 1
'''

def kFoldTest():
    patientList = np.array(['patient01', 'patient02', 'patient03', 'patient04', 'patient05', 'patient06', 'patient07',
                            'patient08', 'patient09', 'patient10', 'patient11', 'patient13', 'patient14', 'patient15',
                            'patient16', 'patient17', 'patient18', 'patient19', 'patient20', 'patient21'])

    kf = KFold(n_splits=15, random_state=None, shuffle=False)
    for train_index, test_index in kf.split(patientList):
        patientTrainData = patientList[train_index]
        patientTestData = patientList[test_index]

filetrain = np.array(['patient01', 'patient02', 'patient03', 'patient04', 'patient05', 'patient06', 'patient07', 'patient08',
                'patient09', 'patient10', 'patient11', 'patient13', 'patient14', 'patient15', 'patient16'])

validatetrain = ['patient17','patient18', 'patient19', 'patient20', 'patient21']

test = dataGenerator(256,256,2,False)#
test2 = dataGenerator(256,256,2,False)

def train_and_predict():
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    #imgs_train, imgs_mask_train = load_train_data()

    #imgs_train = preprocess(imgs_train)
    #imgs_mask_train = preprocess(imgs_mask_train)

    #imgs_train = imgs_train.astype('float32')
    #mean = np.mean(imgs_train)  # mean for data centering
    #std = np.std(imgs_train)  # std for data normalization

    #imgs_train -= mean
    #imgs_train /= std

    #imgs_mask_train = imgs_mask_train.astype('float32')
    #imgs_mask_train /= 2.#255.  # scale masks to [0, 1]
    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet()
    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)

    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    '''
    Batch size defines number of samples that going to be propagated through the network.
    For instance, let's say you have 1050 training samples and you want to set up batch_size equal to 100.
    Algorithm takes first 100 samples (from 1st to 100th) from the training dataset and trains network.
    Next it takes second 100 samples (from 101st to 200th) and train network again. We can keep doing this
    procedure until we will propagate through the networks all samples. The problem usually happens with
    the last set of samples. In our example we've used 1050 which is not divisible by 100 without remainder.
    The simplest solution is just to get final 50 samples and train the network.
    Advantages:
    
    -It requires less memory. Since you train network using less number of samples the overall training
     procedure requires less memory. It's especially important in case if you are not able to fit
     dataset in memory.
    -Typically networks trains faster with mini-batches. That's because we update weights after each
     propagation. In our example we've propagated 11 batches (10 of them had 100 samples and 1 had 50 samples)
     and after each of them we've updated network's parameters. If we used all samples during propagation we
     would make only 1 update for the network's parameter.
     
    Disadvantages:
    -The smaller the batch the less accurate estimate of the gradient. In the figure below you can see that
     mini-batch (green color) gradient's direction fluctuates compare to the full batch (blue color).
     
    https://stats.stackexchange.com/questions/153531/what-is-batch-size-in-neural-network
    '''

    #model.fit(imgs_train, imgs_mask_train, batch_size=5, epochs=300, verbose=1, shuffle=True,
    #           validation_split=0.2,
    #           callbacks=[model_checkpoint])

    #let's say you have a BatchGenerator that yields a large batch of samples at a time
    #(but still small enough for the GPU memory)
    
    model.fit_generator(test.generate(filetrain),validation_data=test2.generate(validatetrain),validation_steps=2400
                        ,steps_per_epoch=12000,epochs=100,verbose=1,callbacks=[model_checkpoint])
    #model.fit(imgs_train, imgs_mask_train, batch_size=5, epochs=2, verbose=1, shuffle=True,
    #              validation_split=0.2,callbacks=[model_checkpoint])

    exit()
    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    imgs_test, imgs_id_test = load_test_data()
    
    imgs_test = preprocess(imgs_test)
    imgs_id_test = preprocess(imgs_id_test)

    #Duong
    sitk.WriteImage(sitk.GetImageFromArray(imgs_test),'C:\\Users\\Duong\\Desktop\\imgs_test.nii.gz')
    sitk.WriteImage(sitk.GetImageFromArray(imgs_id_test),'C:\\Users\\Duong\\Desktop\\imgs_id_test_ground.nii.gz')
    #Duong over

    imgs_test = imgs_test.astype('float32')
    mean = np.mean(imgs_test)  # mean for data centering Duong
    std = np.std(imgs_test)  # std for data normalization Duong
    imgs_test -= mean
    imgs_test /= std

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights('weights.h5')

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    imgs_mask_test = model.predict(imgs_test, verbose=1)
    sitk.WriteImage(sitk.GetImageFromArray(imgs_mask_test), 'C:\\Users\\Duong\\Desktop\\imgs_mask.nii.gz')
    exit()
    np.save('imgs_mask_test.npy', imgs_mask_test)
    print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)
    pred_dir = 'preds'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for image, image_id in zip(imgs_mask_test, imgs_id_test):
        image = (image[:, :, 0] * 255.).astype(np.uint8)
        imsave(os.path.join(pred_dir, str(image_id) + '_pred.png'), image)

if __name__ == '__main__':
    train_and_predict()
