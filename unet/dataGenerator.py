import numpy as np
from skimage.transform import resize
import SimpleITK as sitk

class dataGenerator(object):
    def __init__(self,file_path,dim_x=2,dim_y=2,batch_size=2, shuffle=True):
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.data = 0
        self.segData = 0

        self.__dataSuffix = '_train.npy'
        self.__dataSegSuffix = '_train_mask.npy'

        self.filePath = file_path
        return

    def __get_exploration_order(self, data_list):
        'Generates order of exploration'
        # Find exploration order
        indexes = np.arange(len(data_list))
        if self.shuffle == True:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, data_list):
        'Generates data of batch_size samples'  # X : (n_samples, v_size, v_size, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.dim_x, self.dim_y, 1))
        y = np.empty((self.batch_size, self.dim_x, self.dim_y, 1))

        # Generate data
        for i, slice in enumerate(data_list):
            # Store volume
            #X[i, :, :, 0] = self.data[slice]
            X[i] = self.data[slice]

            # Store class
            #y[i, :, :, 0] = self.segData[slice]
            y[i] = self.segData[slice]

        return X, y

    def __preprocess(self,imgs):
        imgs_p = np.ndarray((imgs.shape[0], self.dim_x, self.dim_y), dtype=np.float)
        for i in range(imgs.shape[0]):
            imgs_p[i] = resize(imgs[i], (self.dim_x, self.dim_y), preserve_range=True)

        imgs_p = imgs_p[..., np.newaxis]
        return imgs_p

    def generate(self, fileList):
        'Generates batches of samples'
        # Infinite loop
        while 1:
            # Generate order of exploration of dataset
            indexes = self.__get_exploration_order(fileList)

            for j in indexes:
                # Resize data
                self.data = self.__preprocess(np.load(self.filePath + fileList[j] + self.__dataSuffix))
                self.segData = self.__preprocess(np.load(self.filePath + fileList[j] + self.__dataSegSuffix))

                mean = np.mean(self.data)  # mean for data centering
                std = np.std(self.data)  # std for data normalization

                self.data -= mean
                self.data /= std

                self.segData /= np.max(self.segData)

                # data [-x => 0] && [x => 0 - 1]
                #minValue = np.min(self.data)
                #self.data += np.abs(minValue)
                #maxValue = np.max(self.data)
                #self.data /= maxValue

                # segmented data [-x => 0] && [x => 0 - 1]
                #minValue = np.min(self.segData)
                #self.segData += np.abs(minValue)
                #maxValue = np.max(self.segData)
                #self.segData /= maxValue

                # Generate batches
                dataLength = self.data.shape[0]

                iMax = int(dataLength / self.batch_size)

                dataSlices = self.__get_exploration_order(np.arange(dataLength))

                for i in range(iMax):
                    # Find slices of file
                    dataSliceTemp = [dataSlices[k]
                                     for k in dataSlices[i*self.batch_size:(i+1)*self.batch_size]]

                    # Generate data
                    X, y = self.__data_generation(dataSliceTemp)
                    yield X, y