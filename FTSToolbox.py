import SimpleITK as sitk
import numpy as np
import os
import scipy.ndimage as ndi
import json
from tqdm import tqdm
from skimage.io import imsave
from scipy.ndimage.morphology import binary_erosion, binary_fill_holes
from skimage.measure import label, marching_cubes_lewiner
import skimage.filters as flts
from skimage.util import img_as_ubyte
from skimage.morphology import disk
from scipy.ndimage.filters import gaussian_filter
import nibabel as nib

class FTSToolbox(object):
    def __init__(self):
        self.data = list() # Important, contains (data, datafilename, data, datafilename ...)
        self.jsonData = list()

        self.dataFormat = '.nii.gz'
        self.fileprefix = 'patient'

        # Parameters
        self.paramCalcMinMax = False
        self.paramSegmentation = False
        self.paramBackgroundThreshold = 0#-600
        self.paramOpCloIteration = 10
        self.paramBackgroundValue = 0#-1000
        self.paramBackgroundSegValue = 0

    # This method loads various data format
    # - json files
    # - nifti files
    # - png files
    def load(self, path):
        '''
        This method loads nifti, json and png files. Loads also several nifti files in a specific folder!
        :param path: Path to specific files or folder!
        :return: Save image data to attribute self.data and related filenames to self.data
        '''
        fileext = path.split(".")
        data = list()
        dataFileName = list()

        # json Files!
        if self.__fileextension(fileext[-1]) == 'json':
            jsonFile = open(path, 'r')
            self.__setJson(json.load(jsonFile))
            return

        # nii / nii.gz / png Files
        elif self.__fileextension(fileext[-1]) == 'nii' or self.__fileextension(fileext[-1]) == 'nii.gz' \
                or self.__fileextension(fileext[-1]) == 'png':
            sitkTempImage = sitk.ReadImage(path)
            data.append(sitkTempImage)
            dataFileName.append(path.split('\\')[-1])

        # if path is a directory
        else:
            fileList = os.listdir(path)

            for fileName in tqdm(fileList):
                self.__fileextension(fileName.split('.')[-1])
                sitkTempImage = sitk.ReadImage(path + fileName)
                data.append(sitkTempImage)
                dataFileName.append(fileName)

        self.__setdata(data)
        self.__setdata(dataFileName)

        return

    # Load Dicom data to RAM! Append data to list
    def loadDICOMData(self, dirPath):
        '''
        This is a specific load routine for dicom files. Please use this method only for dicom files!
        :param dirPath: Directory path to dicom files
        :return: Save image data to attribute self.data and related filenames to self.data
        '''
        dicomDirPaths = self.__getDICOMList(dirPath)
        patientList = list()

        # Init data
        a = 0
        i = 0

        for pathSingle in tqdm(dicomDirPaths):
            pathSplitted = pathSingle.split('\\')
            pathLength = len(pathSplitted)
            patientName = pathSplitted[pathLength - 3]

            # For several CT data per patient
            if patientName in patientList:
                a = a + 1
            else:
                patientList.append(patientName)
                i = i + 1
                a = 0

            sitkImage = self.__dcm2niiConverter(pathSingle)

            # Append data to self.data and self.dataFileName
            data = list()
            dataFileName = list()

            data.append(sitkImage)
            dataFileName.append(self.fileprefix + str(i) + '.' + str(a) + self.dataFormat)

            self.__setdata(data)
            self.__setdata(dataFileName)

            return

    # Save data to harddisk
    def saveToHardDisk(self, outDirPath, datachoice=0):
        '''
        This method is callable if image data is loaded previously. Save data to hard disk.
        :param outDirPath: Output directory path.
        :param datachoice: Optional: Save data or segmentation data? 1 => for segmentation data otherwise data
        :return:
        '''
        self.__checkDataExist()

        # Get data and datafilename
        if datachoice==1:
            data = self.__getdata('segdata')
            dataFileName = self.__getdata('segdatafilename')
        elif datachoice==0:
            data = self.__getdata('data')
            dataFileName = self.__getdata('datafilename')
        else:
            data = self.data[datachoice+2]
            dataFileName = self.data[datachoice+3]

        # Directory exists?
        if not os.path.exists(outDirPath):
            os.makedirs(outDirPath)

        for sitkImage,fileName in tqdm(zip(data,dataFileName)):
            if fileName.find('.npy')>5:
                npImage = sitk.GetArrayFromImage(sitkImage)
                np.save(outDirPath+fileName,npImage)
            elif fileName.find('.png') > 5:
                #self.__saveToPNGFile(sitkImage,outDirPath,fileName)
                npImage = sitk.GetArrayFromImage(sitkImage).astype(float)
                if self.paramSegmentation == False:
                    maskSmaller0 = npImage < 0
                maskGreater0 = npImage >= 0

                # Calculate minimum and maximum value
                #test = len(npImage[maskSmaller0])
                if self.paramSegmentation == False and len(npImage[maskSmaller0]) > 0:
                    minValue = np.min(npImage[maskSmaller0])
                maxValue = np.max(npImage[maskGreater0])

                # Normalize values between -1 and +1
                if self.paramSegmentation == False and len(npImage[maskSmaller0]) > 0:
                    npImage[maskSmaller0] /= np.abs(minValue)
                npImage[maskGreater0] /= maxValue
                imsave(outDirPath+fileName,npImage)
            elif fileName.find('.json') > 5:
                file = open(outDirPath+fileName,'w')
                file.write('{'+sitkImage+'}')
                file.close()
            elif fileName.find('.stl') > 5:
                npImage = sitk.GetArrayFromImage(sitkImage)

                # Use marching cubes to obtain the surface mesh
                verts, faces, normals, values = marching_cubes_lewiner(npImage, 0)

                faces_list = verts[faces]
                normal_list = normals[faces]

                temp_normal_list_sum = np.sum(normal_list,axis=1)

                temp_power_list = np.full_like(temp_normal_list_sum,2)
                temp_power_list = np.power(temp_normal_list_sum, temp_power_list)
                temp_normalize_factor = np.sqrt(np.sum(temp_power_list, 1))
                temp_normalize_factor = np.expand_dims(temp_normalize_factor,1)

                temp_normalize_factor_list = np.concatenate((temp_normalize_factor, temp_normalize_factor, temp_normalize_factor), axis=1)
                facet_normal = temp_normal_list_sum / temp_normalize_factor_list

                file_dump = ''
                file_dump += 'solid test\n'
                for vertex, normal in tqdm(zip(faces_list, facet_normal)):
                    file_dump += 'facet normal ' + str(normal[0]) + ' ' + str(normal[1]) + ' ' + str(normal[2]) + '\n'
                    file_dump += 'outer loop\n'
                    file_dump += 'vertex ' + str(vertex[0][0]) + ' ' + str(vertex[0][1]) + ' ' + str(vertex[0][2]) + '\n'
                    file_dump += 'vertex ' + str(vertex[1][0]) + ' ' + str(vertex[1][1]) + ' ' + str(vertex[1][2]) + '\n'
                    file_dump += 'vertex ' + str(vertex[2][0]) + ' ' + str(vertex[2][1]) + ' ' + str(vertex[2][2]) + '\n'
                    file_dump += 'endloop\n'
                    file_dump += 'endfacet\n'
                file_dump += 'endsolid test'

                f = open(outDirPath+fileName, 'w')
                f.write(file_dump)
                f.close()
            else:
                sitk.WriteImage(sitkImage,outDirPath+fileName)
        return

    def setParameters(self, calcMinMaxArg, segmentationDataOn, paramBackgroundThresholdArg, paramOpCloIterationArg,
                      paramBackgroundValueArg):
        '''
        This method set specific object parameters
        :param calcMinMaxArg: True or false. Calculate ROI?
        :param segmentationDataOn: True or false. Loaded data is segmentation data?
        :param paramBackgroundThresholdArg:
        :param paramOpCloIterationArg: Number of iteration for opening and closing operation
        :param paramBackgroundValueArg:
        :return:
        '''
        self.paramCalcMinMax = calcMinMaxArg
        self.paramSegmentation = segmentationDataOn
        self.paramBackgroundThreshold = paramBackgroundThresholdArg
        self.paramOpCloIteration = paramOpCloIterationArg
        self.paramBackgroundValue = paramBackgroundValueArg

        return

    ''' Image processing methods '''
    # Easy method for data crop
    def cutImage(self):
        '''
        This method uses a dictionary files, where the coordinates of the ROI is declared.
        Json file must be loaded before using this method.
        The following attributes are used in the json-file:
        - zMinROI: Cut from zMinROI in z-direction
        - zMaxROI: Cut to zMaxROI in z direction
        - xMinROI: Cut from xMinROI in x-y plane
        - xMaxROI: Cut to xMaxROI in x-y plane
        - yMinROI: Cut from yMinROI in x-y plane
        - yMaxROI: Cut to yMaxROI in x-y plane
        :return:
        '''
        self.__checkJsonData()
        self.__checkDataExist()

        # Get the right data from list
        data = self.__getdata('data')
        dataFileNameList = self.__getdata('datafilename')

        jsonData = self.__getJson(0)
        dataLength = np.arange(len(dataFileNameList))

        # Convert data to supported format
        self.__convertToSupportedFormat()

        for i in tqdm(dataLength):
            # Get json data z ROI
            zMin = jsonData[dataFileNameList[i]]['zMinROI']
            zMax = jsonData[dataFileNameList[i]]['zMaxROI']

            xMin, yMin, xMax, yMax, sitkTempImage = self.__findMinMaxImagePatch(data[i], zMin, zMax - zMin)

            if self.paramCalcMinMax == False:
                # Get json data x,y ROI
                xMin = jsonData[dataFileNameList[i]]['xMinROI']
                yMin = jsonData[dataFileNameList[i]]['yMinROI']
                zMin = jsonData[dataFileNameList[i]]['zMinROI']
                xMax = jsonData[dataFileNameList[i]]['xMaxROI']
                yMax = jsonData[dataFileNameList[i]]['yMaxROI']
                zMax = jsonData[dataFileNameList[i]]['zMaxROI']

            # Crop image
            #data[i] = self.__imageCrop(sitkTempImage,None, None, yMin, yMax, xMin, xMax)
            data[i] = self.__imageCrop(data[i],zMin, zMax, yMin, yMax, xMin, xMax)

        # Convert data to supported format
        self.__convertToSupportedFormatInverse()

        return

    # Divide image in half
    def bisectImage(self):
        '''
        This method divide the image into half. Use also a dictionary file.
        :return:
        '''
        self.__checkJsonData()
        self.__checkDataExist()

        # Get the right data from list
        data = self.__getdata('data')
        dataFileName = self.__getdata('datafilename')

        jsonData = self.__getJson(0)

        dataLength = np.arange(len(dataFileName))


        for i in tqdm(dataLength):
            xHalf = jsonData[dataFileName[i]]['xHalf']
            xMinROI = jsonData[dataFileName[i]]['xMinROI']

            # Split fileName
            fileNameSplitted = dataFileName[i].split('.')[:-2]
            fileNameLeft = np.append(fileNameSplitted,'left')
            newFileLeft = '.'.join(fileNameLeft)
            fileNameRight = np.append(fileNameSplitted, 'right')
            newFileRight= '.'.join(fileNameRight)

            # Left side
            sitkImageLeft = self.__imageCrop(data[i],xEndArg=xHalf-xMinROI)
            dataFileName[i] = newFileLeft+self.dataFormat

            # Right side
            sitkImageRight = self.__imageCrop(data[i],xStartArg=xHalf-xMinROI + 1)

            # Save Data
            data[i] = sitkImageLeft
            data.append(sitkImageRight)
            dataFileName.append(newFileRight + self.dataFormat)

        return

    # Mirror image
    def mirrorImage(self):
        '''
        This method mirrors the images.
        :return:
        '''
        self.__checkDataExist()

        # Get the right data from list
        data = self.__getdata('data')
        dataFileName = self.__getdata('datafilename')

        dataLength = np.arange(len(dataFileName))

        for i in tqdm(dataLength):
            fileNameSplitted = dataFileName[i].split('.')[0:-2] # Without .nii.ngz
            fileNameSplitted = np.append(fileNameSplitted,'Mirror') # After that we get something like filename.Mirror.nii.gz
            newFileName = '.'.join(fileNameSplitted)
            sitkTempImage = self.__imageMirroring(data[i])

            # Save Data
            data.append(sitkTempImage)
            dataFileName.append(newFileName+self.dataFormat)

        return

    # Traverse through all images and find the max shape.
    def findMaxShape(self):
        '''
        This method finds the max shape of all image data
        :return:
        '''
        self.__checkDataExist()

        # Get data
        data = self.__getdata('data')

        # Preallocation
        shapeList = list()

        for sitkData in tqdm(data):
            npImage = sitk.GetArrayFromImage(sitkData)
            shapeList.append(npImage.shape)

        tempShapeList = np.transpose(shapeList)
        yMax = np.max(tempShapeList[1])
        xMax = np.max(tempShapeList[2])
        return xMax, yMax

    # Convert all images to the same xMaxShape, yMaxShape size
    def convertImgToDefSize(self, xMaxShape, yMaxShape):
        '''
        Convert all images to the same xMaxShape, yMaxShape size
        :param xMaxShape: Size in x
        :param yMaxShape: Size in y
        :return:
        '''
        self.__checkDataExist()

        # Get data
        data = self.__getdata('data')
        dataFileName = self.__getdata('datafilename')

        self.__convertToSupportedFormat()

        dataLength = np.arange(len(dataFileName))

        for i in tqdm(dataLength):
            sitkImage = self.__convertImageToSameSize(data[i], xMaxShape, yMaxShape)
            data[i] = sitkImage

        self.__convertToSupportedFormatInverse()
        return

    # This method assigns the data and data file name into the attribute
    def setDataAttribute(self,list,datachoice):
        '''
        This method assigns the data which is in the list and data file name into the attribute
        :param list: A new list, which will be replaced with saved one.
        :param datachoice: String, data, datafilename, segdata, segdatafilename
        :return:
        '''
        if datachoice == 'data' and len(self.data) >= 1:
            self.data[0] = list
        if datachoice == 'datafilename' and len(self.data) >= 2:
            self.data[1] = list
        if datachoice == 'segdata' and len(self.data) >= 3:
            self.data[2] = list
        if datachoice == 'segdatafilename' and len(self.data) >= 4:
            self.data[3] = list
        return

    # Perform warp and getWarpingMatrix methods on data set
    def elasticDeformation(self, height, width,sigma, alpha):
        '''
        Perform warp and getWarpingMatrix methods on data set. Elastic deformation on a data set.
        :param height: Height from the image. A dataset reveils always the same height.
        :param width: Width from the image. A dataset reveils always the same width.
        :param sigma: Intensity of the displacement.
        :param alpha: Intensity of the wobbling.
        :return:
        '''
        self.__checkDataExist()

        data = self.__getdata('data')
        dataSeg = self.__getdata('segdata')

        i = 0
        warpM = self.__getWarpingMatrix(height,width,sigma,alpha)

        for dataSingle, segSingle in tqdm(zip(data,dataSeg)):
            # Load single image
            npImage = sitk.GetArrayFromImage(dataSingle)
            npImageSeg = sitk.GetArrayFromImage(segSingle)

            # Warp image
            outImage = self.__warp(npImage,warpM)
            outImageSeg = self.__warp(npImageSeg,warpM)

            # Save to data list
            data[i] = sitk.GetImageFromArray(outImage)
            dataSeg[i] = sitk.GetImageFromArray(outImageSeg)
            i = i + 1

        return

    # Merge CT data, top and bottom records
    def mergeData(self):
        '''
        Merge two data sets. Shape of two data sets must be same due logical OR operation.
        :return:
        '''
        self.__checkDataExist()

        # Get data
        data = self.__getdata('data')
        dataFileName = self.__getdata('datafilename')

        # Initialization
        shapeInfo = sitk.GetArrayFromImage(data[0]).shape


        if self.dataFormat == '.npy':
            tempData = np.empty((len(data), shapeInfo[0], shapeInfo[1], shapeInfo[2]))

            fileName = dataFileName[0].replace(self.dataFormat, '')
            for i, sitkImage in enumerate(data):
                tempData[i] = sitk.GetArrayFromImage(sitkImage)

            self.clearData()

            data.append(sitk.GetImageFromArray(tempData))
            dataFileName.append(fileName + self.dataFormat)

        elif self.dataFormat == '.nii.gz':
            dataSeg = self.__getdata('segdata')

            tempData = list()
            tempDataFileName = list()

            for sitkImage1, sitkImage2, fileNameSingle in tqdm(zip(data, dataSeg, dataFileName)):
                # Get indices from self.dataFileName
                npImg1 = sitk.GetArrayFromImage(sitkImage1)
                npImg2 = sitk.GetArrayFromImage(sitkImage2)

                # z shape must be equal!
                if npImg1.shape[0] != npImg2.shape[0]:
                    raise Exception('Shape is not equal!, Img1: ' + npImg1.shape[0] + ' Img2: ' + npImg2.shape[0])

                outImg = np.bitwise_or(npImg1, npImg2)

                # Create fileName
                fileNameSingle = fileNameSingle.replace(self.dataFormat,'')
                splittedFileName = fileNameSingle.split('.')
                newFileName = '.'.join(splittedFileName[:1]) + '.Merged.' + '.'.join(splittedFileName[2:])

                # Append data
                tempData.append(sitk.GetImageFromArray(outImg.astype(int)))
                tempDataFileName.append(newFileName+self.dataFormat)

            self.setDataAttribute(tempData,'data')
            self.setDataAttribute(tempDataFileName,'datafilename')
        return

    def mergeDataRAW(self, nibDirTop, nibDirBot):
        '''
        Merge a list of top data with bottom data
        :param nibDirTop: Directory which contains top data. Important: Filename has the form xx.top.xx
        :param nibDirBot: Directory which contains bottom data. Important: Filename has the form xx.bot.xx
        :return:
        '''
        self.__checkDataExist()

        dataTopfilename = self.__getdata('datafilename')
        dataTop = self.__getdata('data')
        dataBotfilename = self.__getdata('segdatafilename')
        dataBot = self.__getdata('segdata')

        for i, sitkImage in enumerate(dataTop):
            # Check filename if top or bottom and load data
            if dataTopfilename[i].find('top') > 2 and dataBotfilename[i].find('bot') > 2:
                imageTop = dataTop[i]
                imageBot = dataBot[i]
                nibTop = nib.load(nibDirTop+ dataTopfilename[i])
                nibBot = nib.load(nibDirBot+ dataBotfilename[i])
            else:
                raise Exception('filename must contain xxx.top.xxx and xxx.bot.xxx')

            # Collect header information
            headerBottom = nibBot.header
            headerTop = nibTop.header

            # Get voxel spacing in x,y,z
            voxSpacingX = np.array([headerBottom['pixdim'][1], headerTop['pixdim'][1]])
            voxSpacingY = np.array([headerBottom['pixdim'][2], headerTop['pixdim'][2]])
            voxSpacingZ = np.array([headerBottom['pixdim'][3], headerTop['pixdim'][3]])

            # Calculate offset in x,y,z
            xOffset = np.array([headerBottom['qoffset_x'], headerTop['qoffset_x']])
            yOffset = np.array([headerBottom['qoffset_y'], headerTop['qoffset_y']])
            zOffset = np.array([headerBottom['qoffset_z'], headerTop['qoffset_z']])

            xPadding = int((xOffset[1] - xOffset[0]) / voxSpacingX[1])
            yPadding = int((yOffset[1] - yOffset[0]) / voxSpacingY[1])

            # Get array
            npArrayBottom = sitk.GetArrayFromImage(imageBot)

            if self.paramSegmentation:
                backgroundValue = self.paramBackgroundSegValue
            else:
                backgroundValue = self.paramBackgroundValue

            # Pad some backgroundValue in array
            if xPadding < 0:
                npArrayBottom = np.pad(npArrayBottom, ((0, 0), (0, 0), (0, np.abs(xPadding))), 'constant',
                                       constant_values=backgroundValue)[:, :, np.abs(xPadding):]
            else:
                npArrayBottom = np.pad(npArrayBottom, ((0, 0), (0, 0), (np.abs(xPadding) , 0)), 'constant',
                                       constant_values=backgroundValue)[:, :, :npArrayBottom.shape[2]]
            if yPadding < 0:
                npArrayBottom = np.pad(npArrayBottom, ((0, 0), (0, np.abs(yPadding)), (0, 0)), 'constant',
                                       constant_values=backgroundValue)[:, np.abs(yPadding):, :]
            else:
                npArrayBottom = np.pad(npArrayBottom, ((0, 0), (np.abs(yPadding), 0), (0, 0)), 'constant',
                                       constant_values=backgroundValue)[:, :npArrayBottom.shape[1], :]

            # Calculate total size of array and start of top image
            zBeginBot = int(np.ceil((zOffset[1]-zOffset[0])/voxSpacingZ[0]))

            tempdata = list()
            tempdatafilename = list()

            # Get intersected part the bottom data
            tempdata.append(sitk.GetImageFromArray(npArrayBottom[:zBeginBot]))
            if self.paramSegmentation:
                tempdatatemp = list()
                tempdatatempfilename = list()
                diffSlices = npArrayBottom.shape[0]-zBeginBot
                npArrayTop = sitk.GetArrayFromImage(imageTop)

                # Get intersected part of the top data
                tempdata.append(sitk.GetImageFromArray(npArrayTop[diffSlices:]))
                tempdatatemp.append(sitk.GetImageFromArray(npArrayTop[:diffSlices]))
                tempdatatemp.append(sitk.GetImageFromArray(npArrayBottom[zBeginBot:]))
                tempdatafilename.append(dataBotfilename[i])
                tempdatafilename.append(dataTopfilename[i])
                tempdatatempfilename.append('_middle_top.nii.gz')
                tempdatatempfilename.append('_middle_bot.nii.gz')
                self.setDataAttribute(tempdatafilename,'segdatafilename')
                self.setDataAttribute(tempdatatemp,'data')
                self.setDataAttribute(tempdatatempfilename,'datafilename')

            self.setDataAttribute(tempdata,'segdata')

        return

    # Concatenate CT or segmentation data to one big array
    def concToArray(self, filenamedata='dump.npy', filenameseg='dumpseg.npy'):
        '''
        This method concatenate all dataset to one big array. Load a list of file at first!
        :param filenamedata: optional parameter, filename for output file
        :param filenameseg: optional parameter, filename for output file
        :return:
        '''
        data = self.__getdata('data')
        segdata = self.__getdata('segdata')

        height = sitk.GetArrayFromImage(data[0]).shape[1]
        width = sitk.GetArrayFromImage(data[0]).shape[2]

        tempNpData = np.zeros((1, width, height))
        tempNpDataSeg = np.zeros((1, width, height))

        for sitkImage, sitkSegImage in zip(data, segdata):
            npImage = sitk.GetArrayFromImage(sitkImage)
            npSegImage = sitk.GetArrayFromImage(sitkSegImage)

            tempNpData = np.append(tempNpData, npImage, axis=0)
            tempNpDataSeg = np.append(tempNpDataSeg, npSegImage, axis=0)

        tempDataList = list()
        tempSegDataList = list()
        tempDataFilenameList = list()
        tempSegDataFilenameList = list()

        tempDataList.append(sitk.GetImageFromArray(tempNpData))
        tempSegDataList.append(sitk.GetImageFromArray(tempNpDataSeg))
        tempDataFilenameList.append(filenamedata)
        tempSegDataFilenameList.append(filenameseg)

        self.setDataAttribute(tempDataList,'data')
        self.setDataAttribute(tempDataFilenameList,'datafilename')
        self.setDataAttribute(tempSegDataList,'segdata')
        self.setDataAttribute(tempSegDataFilenameList,'segdatafilename')

        return

    # Prepare data for unet. All data appened to one array
    def prepareDataForUNET_old(self, outHeight, outWidth, dataFileName, dataSegFileName):
        # Check if data and json file loaded!
        self.__checkDataExist()
        self.__checkJsonData()

        # Load data
        dataList = self.__getdata('data')
        dataFileNameList = self.__getdata('datafilename')
        jsonData = self.__getJson(0)

        # Preallocation
        tempSegData = list()
        tempSegFileName = list()
        tempNpData = np.zeros((1, outHeight, outWidth))
        tempNpDataSeg = np.zeros((1, outHeight, outWidth))

        if self.checkSegDataExist():
            # Load data
            dataSegList = self.__getdata('segdata')
            dataSegFileNameList = self.__getdata('segdatafilename')

            #Preallocation
            tempzStartEnd = list()

            # Check order if segmentation data exist
            for fileName in dataSegFileNameList:
                fileNameSeg = jsonData[fileName]['memberSegData']
                dataSegIndex = dataSegFileNameList.index(fileNameSeg)

                tempSegData.append(dataSegList[dataSegIndex])
                tempSegFileName.append(dataSegFileNameList[dataSegIndex])

            dataSegFileNameList = tempSegFileName
            dataSegList = tempSegData

            # Fill segmentation numpy array
            for imgSingle in tqdm(dataSegList):
                npImage = sitk.GetArrayFromImage(imgSingle)

                # Save only ROI
                zStart, zEnd = self.getSegzStartEnd(imgSingle)
                tempzStartEnd.append((zStart, zEnd))

                tempNpDataSeg = np.append(tempNpDataSeg, npImage[zStart:zEnd], axis=0)

            # Fill data numpy array
            for imgSingle,zROI in tqdm(zip(dataList,tempzStartEnd)):
                npImage = sitk.GetArrayFromImage(imgSingle)
                tempNpData = np.append(tempNpData, npImage[zROI[0]:zROI[1]], axis=0)

            # Clear data list
            self.clearData()

            tempNpDataSeg = np.delete(tempNpDataSeg, 0, 0)
        else:
            # Fill data numpy array
            for imgSingle in tqdm(dataList):
                npImage = sitk.GetArrayFromImage(imgSingle)
                tempNpData = np.append(tempNpData, npImage, axis=0)

            # Clear data list
            self.clearData()

        tempNpData = np.delete(tempNpData, 0, 0)

        self.__setdata(list(sitk.GetImageFromArray(tempNpData)))
        self.__setdata(list(dataFileName))
        self.__setdata(list(sitk.GetImageFromArray(tempNpDataSeg)))
        self.__setdata(list(dataSegFileName))
        return

    # Clear all data
    def clearData(self):
        '''
        This method flush the memory!
        :return:
        '''
        del self.data[:]
        return

    # Find start/end of a segmentation set
    def getSegzStartEnd_old(self, sitkImageInput):
        # Convert image
        npImage = sitk.GetArrayFromImage(sitkImageInput)

        zStart = 0
        zEnd = npImage.shape[0]

        indicatorStart = False
        indicatorEnd = False

        # Traverse the slices
        for zIndex in range(0, npImage.shape[0]):
            # Sum up all pixels!
            tempSum = np.sum(npImage[zIndex] != 0)

            # Find start of segmentation in z-direction
            if tempSum > 100 and indicatorStart == False:
                zStart = zIndex
                indicatorStart = True

            # Find end of segmentation in z-direction
            if tempSum < 10 and indicatorStart == True and indicatorEnd == False:
                zEnd = zIndex
                indicatorEnd = True
                break

        return zStart, zEnd

    # Normalize image values between 0 and 1
    def normalize(self, img):
        '''
        normalizes the values of img so they are between 0 and 1
        :param img: image
        :return: normalized image
        '''
        return (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-07)

    # Simple method for correcting the values in segmented data
    def adjustImageData(self,img):
        '''
        Arbitrary data values are normalize to 1 .... x
        :param img: a simple ITK image which will be normalize
        :return: a Simple ITK image
        '''
        npImage = sitk.GetArrayFromImage(img).astype(int)
        dataValues = np.bincount(npImage[npImage != 0])
        i = 1
        for dataValue in np.argwhere(dataValues > 10):
            npImage[npImage == dataValue] = i
            i = i + 1

        sitkImage = sitk.GetImageFromArray(npImage.astype(int))
        return sitkImage

    # Convert values in segmentation images in 1 and 2
    def adjustData(self):
        '''
        Convert values in segmentation images in 1 and 2
        :return:
        '''
        self.__checkDataExist()
        data = self.__getdata('data')
        dataFileName = self.__getdata('datafilename')
        if self.paramSegmentation:
            for sitkImage, fileName in tqdm(zip(data,dataFileName)):
                outImage = self.adjustImageData(sitkImage)
                data.append(outImage)
                dataFileName.append(fileName)
        else:
            raise Exception('Please set parameter self.paramSegmentation!')

        return

    # Swap segmented area
    def swapSegData(self,segArea1Value=1,segArea2Value=2):
        '''
        Swap segmented area. Parameter can also use float values, but values are rounding down.
        :param segArea1Value: First value which will be exchange with the second value
        :param segArea2Value: Second value which will be exchange with the first value
        :return:
        '''
        self.__checkDataExist()

        data = self.__getdata('data')

        tempDataList = list()
        for sitkImage in data:
            tempSitkImage = self.__swapSegArea(sitkImage,segArea1Value,segArea2Value)
            tempDataList.append(tempSitkImage)

        self.setDataAttribute(tempDataList,'data')
        return

    # Print shape of each image
    def printShape(self):
        '''
        For debugging purpose. Print shapes of all loaded images
        :return:
        '''
        self.__checkDataExist()

        data = self.__getdata('data')
        dataFileName = self.__getdata('datafilename')

        for sitkImage, fileName in zip(data, dataFileName):
            tempNp = sitk.GetArrayFromImage(sitkImage)
            print(fileName+'&'+str(tempNp.shape[0]))

    # Converts each slices in png
    def Nifti2PNG(self):
        '''
        Split each slice in single images.
        :return:
        '''
        # Filename list
        tempFileName = list()

        # Load data
        data = self.__getdata('data')
        dataFileName = self.__getdata('datafilename')

        # Create a new filename list
        self.dataFormat = '.nii.gz'
        for fileName in dataFileName:
            self.checkFileFormat(fileName)

            # Seperate filename and cut file extension
            fileNameSeperated = fileName.split('.')[:-2] # Remove .nii.gz
            fileNameJoined = '.'.join(fileNameSeperated)

            tempFileName.append(fileNameJoined)

        dataFileName = tempFileName

        # Allocate temp data
        tempData = list()
        tempFileName = list()

        # Split data into single slices
        for sitkImage, fileName in zip(data, dataFileName):
            npImage = sitk.GetArrayFromImage(sitkImage)
            npImage = npImage.astype(float)  # Conversion is very important

            # Traverse the slices
            for zIndex in np.arange(npImage.shape[0]):
                # Exception for exceeding array
                if zIndex >= npImage.shape[0]:
                    Warning('zIndex exceeds array size! But don\'t worry!')
                    break;

                tempData.append(sitk.GetImageFromArray(npImage[zIndex]))
                tempFileName.append(str(zIndex) + '.' + fileName+'.png')

        self.setDataAttribute(tempData,'data')
        self.setDataAttribute(tempFileName,'datafilename')
        return

    # Converts Nifti files in 3D STL files
    def Nifti2STL(self):
        '''
        Converts Nifti files in 3D STL files
        :return:
        '''
        datafilename = self.__getdata('segdatafilename')

        templist = list()
        for filename in datafilename:
            templist.append(filename.replace(self.__fileextension(filename.split('.')[-1]),self.__fileextension('stl')))

        self.setDataAttribute(templist,'segdatafilename')

        return

    # Converts a set of pngs to nifti files
    def PNG2Nifti(self, numberOfDigits=4):
        '''
        Each filename includes a prefix with a number 0012.xxxxx, this number will be sorted
        :param numberOfDigits: A Number of digits for prefix
        :return:
        '''
        # Load data
        data = self.__getdata('data')
        dataFileName = self.__getdata('datafilename')

        # Format must be 1234.patient1....
        self.dataFormat = '.png'
        tempdataFileName = list()
        for fileName in dataFileName:
            self.checkFileFormat(fileName)
            fileNameSplitted = fileName.split('.')
            # Add some additional 0 digits, for a better sort
            digits = fileNameSplitted[0]

            fileNameSplitted[0] = self.__prefixSort(digits, numberOfDigits)

            modifiedFileName = '.'.join(fileNameSplitted)

            tempdataFileName.append(modifiedFileName)

        dataFileName = tempdataFileName

        # Sort list!
        dataFileName, data = zip(*sorted(zip(dataFileName, data)))

        # Get Shape
        imagesetting = sitk.GetArrayFromImage(data[0])
        fileName = '.'.join(dataFileName[0].split('.')[1:-1])

        # Allocate Memory
        tempArray = np.zeros((len(data),imagesetting.shape[0],imagesetting.shape[1]))

        # Merge to one array
        i = 0
        for sitkImage in data:
            npImage = sitk.GetArrayFromImage(sitkImage)
            tempArray[i] = self.__rgb2gray(npImage)

            i += 1

        self.setDataAttribute(list([sitk.GetImageFromArray(tempArray)]),'data')
        self.setDataAttribute(list([fileName+'.nii.gz']),'datafilename')

        return

    def postProcessing(self, numberArtefactsLimit=100, binary_iteration=False,threshold=0.6,twoclass=False):
        '''
        Post processing routine. This routine removes artefacts, fill holes and converts the image to binary image.
        :param numberArtefactsLimit: Gives a threshold of pixel which will be artefacts or not.
        :param binary_iteration: True or false. Fill holes in segmented data?
        :param threshold: Threshold between [0,1].
        :param twoclass: True or false. Segmented data is a two class case?
        :return:
        '''
        self.__checkDataExist()

        # Load data
        data = self.__getdata('data')

        tempData = list()

        for sitkImage in data:
            npImage = sitk.GetArrayFromImage(sitkImage)

            npImage[npImage>threshold] = 1
            npImage = npImage.astype(int)

            # Traverse through all slices
            for sliceCounter in np.arange(npImage.shape[0]):
                # Label connected region
                npImageOut = label(npImage[sliceCounter])
                npImageCounter = np.bincount(npImageOut.flatten())<numberArtefactsLimit

                if twoclass==True and len(npImageCounter)==3:
                    countDataValue = np.bincount(npImageOut.flatten())
                    idx = np.argmin(countDataValue)
                    npImage[sliceCounter][npImageOut == idx] = self.paramBackgroundSegValue


                for numberData in np.argwhere(npImageCounter):
                    npImage[sliceCounter][npImageOut==numberData] = self.paramBackgroundSegValue

                if binary_iteration is not False:
                    npImage[sliceCounter] = binary_fill_holes(npImage[sliceCounter])

            npImageOut = sitk.GetImageFromArray(npImage)
            tempData.append(npImageOut)

        self.setDataAttribute(tempData,'data')

    def subtraction(self,threshold=0.6):
        '''
        Subtract one region with other region. Threshold for converting the image to binary.
        :param threshold:
        :return:
        '''
        self.__checkDataExist()

        data = self.__getdata('data')
        dataSeg = self.__getdata('segdata')

        tempData = list()

        for sitkFullArea,sitkInnerArea in tqdm(zip(data,dataSeg)):
            npFullArea = sitk.GetArrayFromImage(sitkFullArea)
            npInnerArea = sitk.GetArrayFromImage(sitkInnerArea)

            npFullArea = npFullArea-npInnerArea
            npFullArea[npFullArea<(threshold*-1)] = 1
            npFullArea[npFullArea>threshold] = 1

            #npFullArea[npFullArea == npInnerArea] = self.paramBackgroundSegValue

            tempData.append(sitk.GetImageFromArray(npFullArea.astype(int)))

        self.setDataAttribute(tempData,'data')

        return

    def createkDimImages(self, channel, outformat=None):
        '''
        Create an temporary image with channel-dimension. The data_list must be in this form:
        data_list: list((channel1list(...),channel2list(...),channel3list(...)))
        :param channel: Number of channel
        :param outformat: Can be png, nii.gz
        :return:
        '''

        self.__checkDataExist()

        # Load data
        data_list = list()
        dataFileName = self.__getdata('datafilename')

        for i in np.arange(0,channel*2,2):
            data_list.append(self.data[i])

        if outformat == None:
            raise Exception('outformat is not set! Supported format png, nii.gz.')
        # Check if number of list in data_list is the same as channel
        if len(data_list)!=channel:
            raise Exception('Number of data is not divisible by ' + str(channel))

        tempDataList = list()
        tempFileNameList = list()
        zLength = len(data_list[0])

        # Convert sitk-images to array
        for k in np.arange(zLength):
            for i in np.arange(channel):
                data_list[i][k] = sitk.GetArrayFromImage(data_list[i][k])


        for i in np.arange(zLength):
            # Generating tuple
            shape = list()
            shape.append(channel)
            for x in data_list[0][i].shape: shape.append(x)
            shape = tuple(shape)

            # Allocate memory for TempArray
            dataTemp = np.empty(shape)

            for chIndex in np.arange(channel):
                dataTemp[chIndex] = data_list[chIndex][i]

            # Can be upgraded arbitrary
            if outformat == 'png':
                # Decompose into single images
                for j in np.arange(shape[1]):
                    dataSingleTemp = np.empty((shape[0],shape[2],shape[3]))
                    dataSingleTemp[0] = dataTemp[0][j]
                    dataSingleTemp[1] = dataTemp[1][j]
                    dataSingleTemp[2] = dataTemp[2][j]
                    tempFileNameList.append(str(j)+'.'+dataFileName[i].replace('.nii.gz', '.png'))
                    tempDataList.append(sitk.GetImageFromArray(dataSingleTemp.swapaxes(0,2).swapaxes(0,1)))

        self.setDataAttribute(tempDataList,'data')
        self.setDataAttribute(tempFileNameList,'datafilename')

        return

    def otsuThresholding(self,morphClosingIteration=4):
        '''
        Otsu thresholding!
        :param morphClosingIteration: Number of iteration for morphology closing.
        :return:
        '''
        tempList = list()

        # Load data
        data = self.__getdata('data')

        for sitkImage in tqdm(data):
            npImage = sitk.GetArrayFromImage(sitkImage)
            lengthZ = npImage.shape[0]

            tempOutput = np.zeros_like(npImage, dtype=int)

            for sliceIdx in range(lengthZ):
                singleNpImage = npImage[sliceIdx]
                threshold = flts.threshold_otsu(singleNpImage[singleNpImage > self.paramBackgroundValue])
                labelSlice = ndi.measurements.label(singleNpImage > threshold)[0]
                labelsCount = np.bincount(labelSlice[labelSlice != 0])

                # Choose intensity which appears more than 100 times in the resulting image.
                labelChosen = np.argwhere(labelsCount > 100)

                for labelIdx in labelChosen:
                    tempOutput[sliceIdx, :, :] = np.logical_or(tempOutput[sliceIdx, :, :],
                                                               ndi.morphology.binary_closing(labelSlice == labelIdx,
                                                                                             iterations=morphClosingIteration))
            tempList.append(sitk.GetImageFromArray(tempOutput))

        self.setDataAttribute(tempList,'data')
        return

    def otsuThresholdingLocal(self):
        '''
        Local Otsu thresholding
        :return:
        '''
        selem = disk(1)
        tempList = list()

        data = self.__getdata('data')

        for sitkImage in tqdm(data):
            npImage = sitk.GetArrayFromImage(sitkImage).astype(float)
            npImage = npImage/np.max(npImage)
            img = img_as_ubyte(npImage)
            img[img<100] = 0
            local_otsu = flts.rank.otsu(img,selem)

            tempList.append(sitk.GetImageFromArray(local_otsu))
        self.setDataAttribute(tempList,'data')
        return

    def removeArtifacts(self, gaussian_sigma=0.2):
        tempList = list()
        data = self.__getdata('data')
        for sitkImage in tqdm(data):
            # Fetch a patient
            npImage = sitk.GetArrayFromImage(sitkImage)

            # Filter area 1 and 2
            npInnerArea = (npImage == 1).astype(float)
            npOuterArea = (npImage == 2).astype(float)

            # Reserve variables
            tempNpInnerArea = np.zeros_like(npImage)
            tempNpOuterArea = np.zeros_like(npImage)
            tempOuterAreaSmaller = np.zeros_like(npImage)

            i = 0
            for singleInnerSlice, singleOuterSlice in zip(npInnerArea, npOuterArea):
                # Filter inner area
                tempResult = gaussian_filter(singleInnerSlice, gaussian_sigma)
                tempResult = binary_erosion(tempResult, iterations=1)
                tempResult = binary_fill_holes(tempResult)
                tempNpInnerArea[i] = tempResult

                # Filter outer area
                tempResult = gaussian_filter(singleOuterSlice, gaussian_sigma)
                tempResult = binary_erosion(tempResult, iterations=1)
                tempResult = binary_fill_holes(tempResult)
                tempNpOuterArea[i] = tempResult

                tempOuterAreaSmaller[i] = binary_erosion(tempResult, iterations=3)

                # Create Intersection of tempOuterAreaSmaller and tempNpInnerArea
                tempNpInnerArea[i][np.logical_not(tempOuterAreaSmaller[i] == tempNpInnerArea[i])] = 0

                i += 1

            # Merge area 1 and 2
            tempResult = tempNpInnerArea+tempNpOuterArea
            tempList.append(sitk.GetImageFromArray(tempResult.astype(int)))
        self.setDataAttribute(tempList,'data')

    def findMinMaxImagePatch(self):
        segdata = self.__getdata('segdata')
        templist = list()
        for sitksegdata in tqdm(segdata):
            npsegdata = sitk.GetArrayFromImage(sitksegdata)
            scanline = np.ones((npsegdata.shape[1])).astype(int)
            z_list = list()
            x_min = list()
            y_min = list()
            x_max = list()
            y_max = list()
            for z in np.arange(npsegdata.shape[0]):
                for x in np.arange(npsegdata.shape[2]):
                    tempdata = npsegdata[z,:,x]
                    tempresult = np.logical_and(tempdata,scanline)
                    if np.max(tempresult) == True:
                        z_list.append(z)
                        x_min.append(x)
                        break
                for y in np.arange(npsegdata.shape[1]):
                    tempdata = npsegdata[z, y, :]
                    tempresult = np.logical_and(tempdata, scanline)
                    if np.max(tempresult) == True:
                        y_min.append(y)
                        break
                for x in np.arange(npsegdata.shape[2]-1,0,-1):
                    tempdata = npsegdata[z,:,x]
                    tempresult = np.logical_and(tempdata,scanline)
                    if np.max(tempresult) == True:
                        x_max.append(x)
                        break
                for y in np.arange(npsegdata.shape[1]-1,0,-1):
                    tempdata = npsegdata[z, y, :]
                    tempresult = np.logical_and(tempdata, scanline)
                    if np.max(tempresult) == True:
                        y_max.append(y)
                        break
                tupel = {'z':z_list,'x_min':x_min, 'y_min': y_min, 'x_max': x_max, 'y_max':y_max}
            templist.append(tupel)
        self.__setdata(templist)
        return

    def thresholding(self, threshold=0):
        '''
        Convert an image to binary image
        :param threshold: Threshold
        :return:
        '''
        segdata = self.__getdata('segdata')
        tempList = list()
        for sitkImage in segdata:
            npImage = sitk.GetArrayFromImage(sitkImage)
            npImage[npImage < threshold] = 0
            npImage[npImage >= threshold] = 1
            tempList.append(sitk.GetImageFromArray(npImage.astype(int)))

        self.setDataAttribute(tempList,'data')
        return

    def createJson(self, offsetxmin=0, offsetxmax=0, offsetymin=0, offsetymax=0, mode=None):
        tempdumplist = list()
        tempfilename = list()

        datafilename = self.__getdata('datafilename')

        if mode == 'seperate':
            for filename, roiinfo in zip(datafilename, self.data[4]):
                dumpstr = ''
                for i, z_no in enumerate(roiinfo['z']):
                    xmin = roiinfo['x_min'][i]-offsetxmin
                    xmax = roiinfo['x_max'][i]+offsetxmax
                    ymin = roiinfo['y_min'][i]-offsetymin
                    ymax = roiinfo['y_max'][i]+offsetymax

                    dumpstr += '\"' + str(z_no)+ '.' + filename.split('.')[0] + '.png\": {\"xMinROI\": ' + str(xmin) + ', \"yMinROI\": ' + str(
                        ymin) + ', \"xMaxROI\": ' + str(xmax) + \
                              ', \"yMaxROI\": ' + str(ymax) + ', \"zMinROI\": 0, \"zMaxROI\": 4},'
                tempdumplist.append(dumpstr)
                tempfilename.append(filename.split('.')[0] + '.json')
        else:
            for filename, roiinfo in zip(datafilename,self.data[4]):
                xmin = np.min(roiinfo['x_min'])-offsetxmin
                xmax = np.max(roiinfo['x_max'])+offsetxmax
                ymin = np.min(roiinfo['y_min'])-offsetymin
                ymax = np.max(roiinfo['y_max'])+offsetymax
                zmin = np.min(roiinfo['z'])
                zmax = np.max(roiinfo['z'])

                if xmin<0:
                    xmin = 0
                if ymin<0:
                    ymin = 0

                dumpstr = '\"'+filename+'\": {\"xMinROI\": '+str(xmin)+', \"yMinROI\": '+str(ymin)+', \"xMaxROI\": '+str(xmax)+\
                          ', \"yMaxROI\": '+str(ymax)+', \"zMinROI\": '+str(zmin)+', \"zMaxROI\": '+str(zmax)+'}'

                tempdumplist.append(dumpstr)
                tempfilename.append(filename.split('.')[0]+'.json')

        self.setDataAttribute(tempdumplist,'data')
        self.setDataAttribute(tempfilename,'datafilename')

        return






    # Check file format
    def checkFileFormat(self, fileName):
        if fileName.find(self.dataFormat) == -1:
            raise Exception('File format must be ' + self.dataFormat + ' ! -' + fileName + '-')

    # private method
    # Check file extensions are supported
    def __fileextension(self,extension):
        if len(extension) > 15:
            return ''

        if extension == 'json':
            return 'json'
        elif extension == 'nii':
            return 'nii'
        elif extension == 'png':
            return 'png'
        elif extension == 'gz':
            return 'nii.gz'
        elif extension == 'stl':
            return 'stl'

        raise Exception('This format ' + extension + 'is unknown!')

    # Easy dicom to nifti converter
    def __dcm2niiConverter(self, input_folder):
        """
        This function is a dicom to nifti converter!
        """

        # Create Image Series Reader object
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(input_folder)
        reader.SetFileNames(dicom_names)

        # Start reading files
        imgSITK = reader.Execute()

        return imgSITK

    # Please use getDICOMList, otherwise this function returns a single path! Use getDICOMfolder
    def __getDICOMFolder(self, rootDirectory=None):
        # Init string variable
        outStr = ''

        try:
            # Try access subdirectory!
            subDirectories = os.listdir(rootDirectory)
        except:
            return '\n'

        # Traverse all sub directories
        for subDirStr in subDirectories:
            try:
                # termination condition!
                if 'IM_0' in subDirStr:
                    return rootDirectory

                # Concatenate all paths!
                outStr += self.__getDICOMFolder(rootDirectory + '\\' + subDirStr) + '\n'
            except AttributeError:
                return None

        return outStr

    # Wrapper for getDICOMFolder, to generate a list of paths
    def __getDICOMList(self, rootFolder):
        pathList = list()
        pathStr = self.__getDICOMFolder(rootFolder).split('\n')

        for pathStrAct in pathStr:
            if len(pathStrAct) > 10:
                pathList.append(pathStrAct)

        return pathList

    # Check method for data
    def __checkDataExist(self):
        if len(self.data):
            return True
        else:
            raise Exception('Empty data, please load data at first!')
            return False

    # Easy method for cutting data
    def __imageCrop(self, imgSitkInput, zStartArg=None, zEndArg=None, yStartArg=None, yEndArg=None, xStartArg=None,
                  xEndArg=None):
        imgNP = sitk.GetArrayFromImage(imgSitkInput)

        # Assign Parameter
        z_start = zStartArg
        z_finish = zEndArg
        y_start = yStartArg
        y_finish = yEndArg
        x_start = xStartArg
        x_finish = xEndArg

        # z Test
        if z_start == None or z_start < 0 or z_finish > (imgNP.shape[0] - 1):
            z_start = 0
        if z_finish == None or z_finish > (imgNP.shape[0] - 1):
            z_finish = imgNP.shape[0]

        # y Test
        if y_start == None or y_start < 0 or y_start > (imgNP.shape[1] - 1):
            y_start = 0
        if y_finish == None or y_finish > (imgNP.shape[1] - 1):
            y_finish = imgNP.shape[1]

        # x Test
        if x_start == None or x_start < 0 or x_start > (imgNP.shape[2] - 1):
            x_start = 0
        if x_finish == None or x_finish > (imgNP.shape[2] - 1):
            x_finish = imgNP.shape[2]

        # Write image
        return sitk.GetImageFromArray(imgNP[z_start:z_finish, y_start:y_finish, x_start:x_finish])

    # Find ROI in x,y
    def __findMinMaxImagePatch(self, sitkImageInput, zMinArg=None, zRangeArg=None):
        # define some parameters
        threshold = self.paramBackgroundThreshold
        opCloIteration = self.paramOpCloIteration # number of iteration for binary erosion and dilation
        background = self.paramBackgroundValue
        zLow = zMinArg
        zRange = zRangeArg

        # Get numpy Array
        npImage = np.swapaxes(sitk.GetArrayFromImage(sitkImageInput), 0, 2)

        # Check x,y,z
        if zLow == None or zLow < 0 or zLow > (npImage.shape[2]):
            zLow = 0

        if zRange == None or zRange > (npImage.shape[2]):
            zRange = npImage.shape[2]

        if (zLow + zRange) > npImage.shape[2]:
            zRange = npImage.shape[2] - zLow - 1

        # segmentation data?
        if not self.paramSegmentation:
            # Simple thresholding + erosion and dilation for cutting unnecessary areas
            imageMask = npImage > threshold
            imageMaskEroded = ndi.morphology.binary_erosion(imageMask, iterations=opCloIteration)
            imageMaskExpanded = ndi.morphology.binary_dilation(imageMaskEroded, iterations=opCloIteration + 5)

            # define background
            npImage[np.logical_not(imageMaskExpanded)] = background

            imageMaskExpandedFlat = np.argwhere(imageMaskExpanded[:, :, zLow:(zLow + zRange)].flatten())
            index = np.unravel_index(imageMaskExpandedFlat, imageMaskExpanded[:, :, zLow:(zLow + zRange)].shape)

            yMin = np.min(index[1])
            xMin = np.min(index[0])
            yMax = np.max(index[1])
            xMax = np.max(index[0])
        else:
            xMin, yMin, xMax, yMax = 0,0,0,0

        # Convert to output Simple ITK Image
        sitkImage = sitk.GetImageFromArray(np.swapaxes(npImage[:, :, zLow:(zLow + zRange)].astype(int), 0, 2))
        return xMin, yMin, xMax, yMax, sitkImage

    # Check method for jsonData
    def __checkJsonData(self):
        if len(self.jsonData):
            return True
        else:
            raise Exception('Json File not loaded, please load json file first!')
            return False

    # Append data in to self.data
    def __setdata(self, datalist):
        self.data.append(datalist)
        return

    # This is a specific function for getting data, datafilename or segmentationdata, segmentationdatafilename
    def __getdata(self,datachoice):
        if len(self.data) <= 0:
            raise Exception('self.data is empty! Please load data first!')

        if datachoice == 'data' and len(self.data) >= 1:
            return self.data[0]
        if datachoice == 'datafilename' and len(self.data) >= 2:
            return self.data[1]
        if datachoice == 'segdata' and len(self.data) >= 3:
            return self.data[2]
        if datachoice == 'segdatafilename' and len(self.data) >= 4:
            return self.data[3]

        raise Exception(datachoice +' not loaded! Please load data first!')

    # Append json-file in to self.jsonData
    def __setJson(self, jsondata):
        self.jsonData.append(jsondata)

    # Append json-file in to self.jsonData
    def __getJson(self, jsonchoice):
        return self.jsonData[jsonchoice]

    # Convert dataset to the same shape
    def __convertImageToSameSize(self, sitkImageInput, xMaxArg, yMaxArg):
        # Convert all CTs to the same size
        npImage = np.swapaxes(sitk.GetArrayFromImage(sitkImageInput), 0, 2)

        if self.paramSegmentation:
            constantValues = self.paramBackgroundSegValue
        else:
            constantValues = self.paramBackgroundValue

        xNpImage = npImage.shape[0]
        yNpImage = npImage.shape[1]
        xMax = xMaxArg
        yMax = yMaxArg

        # Do some calculus
        xDifference = float(xMax - xNpImage) / 2
        yDifference = float(yMax - yNpImage) / 2

        # What about smaller images?
        if xDifference < 0:
            xC = 0
            xF = 0
            npImage = npImage[:xMax,:,:]
        else:
            # Needs integer for numpy.pad
            xC = int(np.ceil(xDifference))
            xF = int(np.floor(xDifference))

        if yDifference < 0:
            yC = 0
            yF = 0
            npImage = npImage[:,:yMax,:]
        else:
            # Needs integer for numpy.pad
            yC = int(np.ceil(yDifference))
            yF = int(np.floor(yDifference))

        # Image padding
        paddedImage = np.pad(npImage, ((xC, xF), (yC, yF), (0, 0)), 'constant', constant_values=constantValues)

        if paddedImage.shape[0] != xMax or paddedImage.shape[1] != yMax:
            paddedImage = np.pad(npImage,
                                 ((0, int(xMax - paddedImage.shape[0])), (0, int(yMax - paddedImage.shape[1])), (0, 0)),
                                 'constant', constant_values=constantValues)

        return sitk.GetImageFromArray(np.swapaxes(paddedImage.astype(int), 0, 2))

    # Execute elastic deformation
    def __warp(self, img, warpM):
        '''
        elastically deforms image by applying a displacement given by warpM to each pixel position, near neighbours!
        :param img: original image
        :param warpM: matrix with displacements
        :return: elastically deformed image
        '''
        severalImgs = False
        if img.shape.__len__() > 2:
            if img.shape[2] != 3:
                severalImgs = True

        # calculate warped image
        if severalImgs:
            warpedImg = img[:, warpM[0].astype(int), warpM[1].astype(int)]
        else:
            warpedImg = img[warpM[0].astype(int), warpM[1].astype(int)]

        return warpedImg

    # creates displacement matrix for elastic deformation
    def __getWarpingMatrix(self, rows, cols, sigma, alpha):
        '''
        creates displacement matrix
        :param rows: number of rows
        :param cols: number of cols
        :param sigma: smoothing factor for gaussian smoothing
        :param alpha: maximum displacement
        :return: displacement matrix
        '''
        w = np.random.rand(2, rows, cols)
        w[0] = gaussian_filter(w[0], sigma)
        w[1] = gaussian_filter(w[1], sigma)
        w = (self.normalize(w) - 0.5) * alpha

        # calc indices with transpose
        vec = np.array(range(rows))
        matrix = np.array([vec, ] * cols)
        w[0] += matrix.transpose()
        vec = np.array(range(cols))
        matrix = np.array([vec, ] * rows)
        w[1] += matrix

        # use mirrored image when index out of bounds
        w = np.abs(w)
        w[0][w[0] >= rows] = rows - w[0][w[0] >= rows] % rows - 1
        w[1][w[1] >= cols] = cols - w[1][w[1] >= cols] % cols - 1

        return w

    # Swap segmentation area
    def __swapSegArea(self,img,segArea1Value=None,segArea2Value=None):
        if segArea1Value==None or segArea2Value==None:
            raise ValueError('In function swapSegArea, segArea1Value or segArea2Value not set!')

        npImage = sitk.GetArrayFromImage(img).astype(int)
        npImage[npImage==segArea1Value] = 100
        npImage[npImage==segArea2Value] = segArea1Value
        npImage[npImage==100] = segArea2Value
        return sitk.GetImageFromArray(npImage)

    # Sort prefix
    def __prefixSort(self, digits, numberOfDigits):
        if self.__checkDigit(digits):
            if len(digits) < numberOfDigits:
                for i in np.arange(numberOfDigits - len(digits)):
                    digits = '0' + digits
        else:
            raise Exception('Filename format must be 1234.patientX...., prefix is not a number')

        return digits

    def __rgb2gray(self, img):
        if len(img.shape) <= 2:
            # Do nothing
            return img
        else:
            # https://en.wikipedia.org/wiki/Grayscale
            return np.dot(img[...,:img.shape[2]],[0.299, 0.587, 0.114])

    # Image mirroring
    def __imageMirroring(self, sitkImageInput):
        npImage = sitk.GetArrayFromImage(sitkImageInput)
        npImageFlipped = np.flip(npImage, 2)
        return sitk.GetImageFromArray(npImageFlipped)

    def __checkDigit(self, fileName):
        if fileName.isdigit():
            return True
        else:
            return False

    def __convertToSupportedFormat(self):
        data = self.__getdata('data')
        dataFileNameList = self.__getdata('datafilename')

        # Convert data to supported format
        i = 0
        for sitkImage, filename in zip(data, dataFileNameList):
            if self.__fileextension(filename.split('.')[-1]) == 'png':
                npImage = sitk.GetArrayFromImage(sitkImage)

                # one channel image
                if len(npImage.shape) < 3:
                    npTempImage = np.zeros((npImage.shape[0], npImage.shape[1], 3))
                    npTempImage[:, :, 0] = npImage
                    npTempImage[:, :, 1] = npImage
                    npTempImage[:, :, 2] = npImage
                    npImage = npTempImage

                npImage = npImage.swapaxes(0, 1)
                npImage = npImage.swapaxes(0, 2)
                data[i] = sitk.GetImageFromArray(npImage)
            else:
                break
            i += 1

    def __convertToSupportedFormatInverse(self):
        data = self.__getdata('data')
        dataFileNameList = self.__getdata('datafilename')

        i = 0
        for sitkImage, filename in zip(data, dataFileNameList):
            if self.__fileextension(filename.split('.')[-1]) == 'png':
                npImage = sitk.GetArrayFromImage(sitkImage)
                npImage = npImage.swapaxes(0, 2)
                npImage = npImage.swapaxes(0, 1)
                data[i] = sitk.GetImageFromArray(npImage)
            else:
                break
            i += 1