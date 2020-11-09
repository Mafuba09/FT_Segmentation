from net import *
from sklearn.model_selection import KFold
import os
from skimage.io import imsave

# Neural Network predict
network = net(256,256,2,False)
print('Load Test data')
data = network.loadData('C:\\dump_file.npy')
print('Load Groundtruth')
dataGroundTruth = network.loadData('C:\\dumpseg.npy',True)
network.getUnet()
network.loadWeights('C:\\example_model.h5')
print('Predict data')
predictedData = network.predictNet(data)
network.saveDataAsNifti('C:\\patient_predicted.nii.gz')
exit()


# Neural Network Training
network = net(256,256,2,False)
print('CT data and segmentation data')
ct = network.loadData('C:\\dump_file.npy')
seg = network.loadData('C:\\dumpseg.npy',True)
network.createTrainingValidationDataFactor(ct[0],seg[0],factor=0.2)
network.getUnet()
network.createModelCheckpoint('C:\\test.h5')
print('Train neural network')
network.fitNet(30,1000,100)
exit()