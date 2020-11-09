from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dense, Flatten, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.utils import multi_gpu_model

from skimage.transform import resize
from dataGenerator import *

import numpy as np
import SimpleITK as sitk
import imageio

from keras.preprocessing.image import array_to_img, img_to_array, load_img

# Workaround https://github.com/keras-team/keras/issues/2436#issuecomment-354882296
class ModelMGPU(Model):
    def __init__(self, ser_model, gpus):
        pmodel = multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        '''
        # return Model.__getattribute__(self, attrname)
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(ModelMGPU, self).__getattribute__(attrname)

class net(object):
    ################################################### New Version
    # Public functions
    def __init__(self,dim_x=256,dim_y=256,batch_size=1,shuffle=True,smooth=0,multigpu=1):
        self.__dim_x = dim_x
        self.__dim_y = dim_y
        self.__smooth = smooth
        self.__batchSize = batch_size
        self.__shuffle = shuffle
        self.trainingData = None # Numpy Array
        self.trainingSegData = None # Numpy Array
        self.validationData = None # Numpy Array
        self.validationSegData = None # Numpy Array
        self.monitor_value = 'val_loss'
        self.__dataStore = None
        self.__multigpu = multigpu
        self.__neuronalNetworkModel = None

        self.__dataFileNameList = None
        self.__segFileNameList = None
        self.__dataValidFileNameList = None
        self.__segValidFileNameList = None

        self.__model_checkpoint = None
        return

    def loadData(self,data_filename_arg,segmented_data=False):
        if segmented_data==False and isinstance(data_filename_arg,str):
            print('Notice: Sure that '+ data_filename_arg +' is not segmentation?')
        if not isinstance(data_filename_arg,list):
            dataFileNameList = list([data_filename_arg])
        else:
            dataFileNameList = data_filename_arg

        data_list = list()
        mean_list = list()
        std_list = list()

        # Performance reason, no if/else statements in loops!
        if dataFileNameList[0].find('.npy') > 5:
            for data_filename in dataFileNameList:
                data = np.load(data_filename)
                data = self.__preprocess(data)

                meanstd = data.astype('float32')
                mean_list.append(np.mean(meanstd))
                std_list.append(np.std(meanstd))

                data_list.append(data)

        elif dataFileNameList[0].find('.png') > 5:
            for data_filename in dataFileNameList:
                data = imageio.imread(data_filename)
                # One channel images
                if len(data.shape) < 3:
                    data = data.swapaxes(0, 1)
                    tempdata = np.zeros((3, data.shape[0], data.shape[1]))
                    tempdata[0] = data
                    tempdata[1] = data
                    tempdata[2] = data
                    data = tempdata
                else:
                    data = data.swapaxes(0, 2)

                data = self.__preprocessPNG(data)

                meanstd = data#data.astype('float32')
                mean_list.append(np.mean(meanstd))
                std_list.append(np.std(meanstd))

                data_list.append(data)

        elif dataFileNameList[0].find('.nii.gz') > 5:
            for data_filename in dataFileNameList:
                data = sitk.GetArrayFromImage(sitk.ReadImage(data_filename))
                data = self.__preprocess(data)

                meanstd = data.astype('float32')
                mean_list.append(np.mean(meanstd))
                std_list.append(np.std(meanstd))

                data_list.append(data)

        else:
            raise Exception('Not supported format!')

        if segmented_data == False:
            i = 0
            for data,mean,std in zip(data_list,mean_list,std_list):
                data_list[i] = (data-mean)/std
                i += 1
        else:
            i = 0
            # Segmentation must be between 0 and 1 otherwise losserror
            for data in data_list:
                data_list[i] = data/(np.max(data))
                i += 1

        '''for data_filename in dataFileNameList:
            if data_filename.find('.npy') > 5:
                data = np.load(data_filename)
                data = self.__preprocess(data)

            if data_filename.find('.png') > 5:
                data = imageio.imread(data_filename)
                # One channel images
                if len(data.shape) < 3:
                    data = data.swapaxes(0,1)
                    tempdata = np.zeros((3,data.shape[0],data.shape[1]))
                    tempdata[0] = data
                    tempdata[1] = data
                    tempdata[2] = data
                    data = tempdata
                else:
                    data = data.swapaxes(0, 2)
                data = self.__preprocessPNG(data)

            if data_filename.find('.nii.gz') > 5:
                data = sitk.GetArrayFromImage(sitk.ReadImage(data_filename))
                data = self.__preprocess(data)

            if segmented_data==False:
                data = data.astype('float32')

                mean = np.mean(data)  # mean for data centering Duong
                std = np.std(data)  # std for data normalization Duong

                data -= mean
                data /= std

            data_list.append(data)

        # data [-x => 0] && [x => 0 - 1]
        #minValue = np.min(imgTestData)
        #imgTestData += np.abs(minValue)
        #maxValue = np.max(imgTestData)
        #imgTestData /= maxValue'''
        return data_list

    def concatenateData(self, listOfData):
        if not isinstance(listOfData,(list)):
            raise Exception('listOfData must be a list object!')
        if len(listOfData) < 2:
            raise Exception('Concatentation not possible, only one element!')

        concatenedArray = np.concatenate(tuple(listOfData),axis=0)

        return concatenedArray

    def createTrainingValidationDataFactor(self, npdata, npsegmentation, factor=0.2):
        if npdata.shape != npsegmentation.shape:
            raise Exception('nptrain and npsegmentation must have the same shape!')

        numberOfImages = npdata.shape[0]
        numberOfImagesValidation = int(factor*numberOfImages)

        # Split data in training and validation
        temptrainvalid = np.split(npdata,[numberOfImagesValidation])
        validationData = temptrainvalid[0]
        trainingData = temptrainvalid[1]

        # Split segmentation data in training and validation
        temptrainvalid = np.split(npsegmentation, [numberOfImagesValidation])
        validationSegData = temptrainvalid[0]
        trainingSegData = temptrainvalid[1]

        self.trainingData = trainingData
        self.trainingSegData = trainingSegData
        self.validationData = validationData
        self.validationSegData = validationSegData
        return

    def createTrainingData(self, npdata, npsegmentation):
        self.trainingData = npdata
        self.trainingSegData = npsegmentation
        return

    def createValidationData(self, npdata, npsegmentation):
        self.validationData = npdata
        self.validationSegData = npsegmentation
        return

    def createGenerator(self,dataFileNameList, segFileNameList, dataValidFileNameList, segValidFileNameList):
        self.__dataFileNameList = dataFileNameList
        self.__segFileNameList = segFileNameList
        self.__dataValidFileNameList = dataValidFileNameList
        self.__segValidFileNameList = segValidFileNameList

        return

    def getUnet(self,learning_rate=1e-6):
        inputs = Input((self.__dim_x, self.__dim_y, 1))
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
        if self.__multigpu > 1:
            parallel_model = ModelMGPU(model,gpus=self.__multigpu)#multi_gpu_model(model,gpus=self.__multigpu)
            parallel_model.compile(optimizer=Adam(lr=learning_rate), loss=self.__dice_coef_loss,
                                     metrics=[self.__dice_coef])
            self.__neuronalNetworkModel = parallel_model
        else:
            model.compile(optimizer=Adam(lr=learning_rate), loss=self.__dice_coef_loss, metrics=[self.__dice_coef])
            self.__neuronalNetworkModel = model

        return

    def getUnetPNG(self,learning_rate=1e-6):
        inputs = Input((self.__dim_x, self.__dim_y,1))
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
        conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

        up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

        up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

        up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

        conv10 = Conv2D(2, (1, 1), activation='sigmoid')(conv9)

        model = Model(inputs=[inputs], outputs=[conv10])

        if self.__multigpu > 1:
            parallel_model = ModelMGPU(model,gpus=self.__multigpu)#multi_gpu_model(model,gpus=self.__multigpu)
            parallel_model.compile(optimizer=Adam(lr=learning_rate), loss=self.__dice_coef_loss,
                                     metrics=[self.__dice_coef])
            self.__neuronalNetworkModel = parallel_model
        else:
            model.compile(optimizer=Adam(lr=learning_rate), loss=self.__dice_coef_loss, metrics=[self.__dice_coef])
            self.__neuronalNetworkModel = model

        return

    def fitNetWithGenerator(self,epochs,number_data,number_validation_data, numberOfLoadedData, numberOfLoadedDataValid):
        self.__checkNeuronalNetwork()
        self.__checkModelCheckpoint()
        self.__neuronalNetworkModel.fit_generator(self.generator(self.__dataFileNameList, self.__segFileNameList, numberOfLoadedData),
                                                  validation_data=self.generator(self.__dataValidFileNameList, self.__segValidFileNameList, numberOfLoadedDataValid),
                                                  validation_steps=number_validation_data, steps_per_epoch=number_data,
                                                  epochs=epochs, verbose=1,callbacks=[self.__model_checkpoint])

    def fitNet(self,epochs,number_data,number_validation_data):
        self.__checkNeuronalNetwork()
        self.__checkModelCheckpoint()
        self.__checkGeneratorCreated()
        self.__neuronalNetworkModel.fit(self.trainingData, self.trainingSegData, batch_size=self.__batchSize, epochs=epochs,
                                        callbacks=[self.__model_checkpoint], validation_data=(self.validationData,self.validationSegData))

    def predictNet(self, data):
        self.__checkNeuronalNetwork()
        imgPredictedData = self.__neuronalNetworkModel.predict(data, verbose=1)
        self.__dataStore = imgPredictedData
        return imgPredictedData

    def createModelCheckpoint(self,weight_filename):
        self.__model_checkpoint = ModelCheckpoint(weight_filename, monitor=self.monitor_value, save_best_only=True)

    def loadWeights(self, weight_filename):
        self.__checkNeuronalNetwork()
        self.__neuronalNetworkModel.load_weights(weight_filename)

    def generator(self, dataFileNameList, segFileNameList, numberOfLoadedData):
        if not isinstance(dataFileNameList,(list)) or not isinstance(segFileNameList,(list)):
            raise Exception('dataFileNameList or segFileNameList not a list')

        'Generates batches of samples'
        # Infinite loop
        i = 0
        while 1:
            # Load a numberOfLoadedData
            tempDataList = list()
            tempSegList = list()

            idxStart = i*numberOfLoadedData
            if idxStart > len(dataFileNameList):
                idxStart = len(dataFileNameList)-1
            idxEnd = ((i+1)*numberOfLoadedData)-1
            if idxEnd > len(dataFileNameList):
                idxEnd = len(dataFileNameList)-1
                i = -1

            for dataFileName, segFileName in zip(dataFileNameList[idxStart:idxEnd], segFileNameList[idxStart:idxEnd]):
                tempDataList.append(self.loadData(dataFileName))
                tempSegList.append(self.loadData(segFileName,segmented_data=True))

            self.trainingData = self.concatenateData(tempDataList)
            self.trainingSegData = self.concatenateData(tempSegList)

            zMax = self.trainingData.shape[0]

            # Generate order of exploration of dataset
            indexes = self.__get_exploration_order(np.arange(zMax))

            for j in range(zMax):
                # Find slices of file
                idxTemp = [indexes[k]
                                 for k in indexes[j * self.__batchSize:(j + 1) * self.__batchSize]]

                # Generate data
                X, y = self.__data_generation(idxTemp)
                yield X, y

            i += 1

    def saveDataAsNifti(self, saveFileName):#,saveFileName1,saveFileName2):
        sitk.WriteImage(sitk.GetImageFromArray(self.__dataStore), saveFileName)
        #sitk.WriteImage(sitk.GetImageFromArray(self.__dataStore[:,:,:,0]), saveFileName)
        #sitk.WriteImage(sitk.GetImageFromArray(self.__dataStore[:,:,:,1]), saveFileName1)
        #sitk.WriteImage(sitk.GetImageFromArray(self.__dataStore[:,:,:,2]), saveFileName2)
        return

    def saveDataAspng(self,saveFileFolder):
        i = 0
        for data in self.__dataStore:
            print(np.max(data))
            imageio.imsave(saveFileFolder+str(i)+'.png',data)
            i += 1

    # Private functinos
    def __preprocess(self,imgs):
        imgs_p = np.ndarray((imgs.shape[0], self.__dim_x, self.__dim_y), dtype=np.float)
        for i in range(imgs.shape[0]):
            imgs_p[i] = resize(imgs[i], (self.__dim_x, self.__dim_y), preserve_range=True)

        imgs_p = imgs_p[..., np.newaxis]
        return imgs_p

    def __preprocessPNG(self,imgs):
        imgs_p = np.ndarray((imgs.shape[0], self.__dim_x, self.__dim_y), dtype=np.float)
        for i in range(imgs.shape[0]):
            imgs_p[i] = resize(imgs[i], (self.__dim_x, self.__dim_y), preserve_range=True)

        imgs_p = imgs_p.swapaxes(0,2)

        imgs_p = imgs_p[np.newaxis, ...]

        return imgs_p

    def __dice_coef(self,y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        # F1 Score
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + self.__smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + self.__smooth)

    def __dice_coef_loss(self,y_true, y_pred):
        return -self.__dice_coef(y_true, y_pred)

    def   __hausdorff_distance(self):
        K

    def __checkNeuronalNetwork(self):
        if self.__neuronalNetworkModel == None:
            raise Exception('Neuronal Network not set! Please execute getUnet(...) at first')

    def __checkModelCheckpoint(self):
        if self.__model_checkpoint == None:
            raise Exception('Model checkpoint not set! Please execute createModelCheckpoint(...) at first')

    def __checkGeneratorCreated(self):
        if self.__dataFileNameList != None or self.__segFileNameList != None or self.__dataValidFileNameList != None or self.__segValidFileNameList != None:
            raise Exception('Generator created!')

    # Data generator
    def __get_exploration_order(self, data_list, shuffle=False):
        'Generates order of exploration'
        # Find exploration order
        indexes = np.arange(len(data_list))
        if shuffle == True:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, data_list):
        'Generates data of batch_size samples'  # X : (n_samples, v_size, v_size, n_channels)
        # Initialization
        X = np.empty((self.__batchSize, self.__dim_x, self.__dim_y, 1))
        y = np.empty((self.__batchSize, self.__dim_x, self.__dim_y, 1))

        # Generate data
        for i, slice in enumerate(data_list):
            # Store data
            #X[i, :, :, 0] = self.data[slice]
            X[i] = self.trainingData[slice]

            # Store segmentation
            #y[i, :, :, 0] = self.segData[slice]
            y[i] = self.trainingSegData[slice]

        return X, y