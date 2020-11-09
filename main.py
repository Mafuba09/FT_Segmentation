from FTSToolbox import *

fts_object = FTSToolbox()

# Example for creating STL files
'''fts_object.load('patient1.left-targets.nii.gz') # Load data
fts_object.load('patient1.left-targets.nii.gz') # Load same data
fts_object.Nifti2STL()
fts_object.saveToHardDisk('.\\output_dir\\',1)
exit()'''

# Example for creating dictionary files
'''fts_object.load('patient1.nii.gz') # Load CT data
fts_object.load('patient1.merge.nii.gz') # Load Segmentation data, always in this order
fts_object.findMinMaxImagePatch()
fts_object.createJson(mode=None) # mode = None, determine roi for all slices, mode = seperate, determine roi for each slice
fts_object.saveToHardDisk('.\\output_dir\\')
exit()'''

# Example for getting necessary images only and creating dictionary file
'''fts_object.load('.\\ground\\') # Load CT data or ground data
fts_object.load('.\\ground\\') # Load Segmentation data
fts_object.findMinMaxImagePatch()

# Cache data
tempdata = fts_object.data[0]
tempfilename = fts_object.data[1]
fts_object.createJson(mode='seperate')
fts_object.saveToHardDisk('.\\output_dir\\')

# Restore data
fts_object.data[0] = tempdata
fts_object.data[1] = tempfilename

fts_object.Nifti2PNG()

tempList = list()
tempListName = list()

# Get necessary images
for i, filename in enumerate(fts_object.data[1]):
    if int(filename.split('.')[0]) in fts_object.data[4][0]['z']:
        tempList.append(fts_object.data[0][i])
        tempListName.append(filename)

fts_object.setDataAttribute(tempList,'data')
fts_object.setDataAttribute(tempListName,'datafilename')

fts_object.saveToHardDisk('.\\ground\\')
exit()'''

# Example for image cut with dictionary file
'''fts_object.setParameters(False,True,0,10,0)
fts_object.load('patient1.json')
fts_object.load('.\\input\\')
fts_object.cutImage()
fts_object.saveToHardDisk('.\\output_dir\\')
exit()'''

# Example for Saving data as numpy binary .npy
fts_object.load('.\\input\\')
fts_object.load('.\\ground\\')
fts_object.concToArray()
fts_object.saveToHardDisk('.\\output_dir\\')
fts_object.saveToHardDisk('.\\output_dir\\',1)
exit()

# Example for creating RGB png
'''fts_object.load('patient1.nii.gz')
fts_object.load('patient1.nii.gz')
fts_object.load('patient1.nii.gz')
temp1 = sitk.GetArrayFromImage(fts_object.data[0][0])
temp2 = sitk.GetArrayFromImage(fts_object.data[2][0])
temp3 = sitk.GetArrayFromImage(fts_object.data[4][0])

npTemp1 = np.zeros_like(temp1)
npTemp2 = np.zeros_like(temp2)
npTemp3 = np.zeros_like(temp3)

npTemp1[:] = temp1[:]
npTemp2[:-1] = temp2[1:]
npTemp3[:-2] = temp3[2:]

fts_object.data[0][0] = sitk.GetImageFromArray(npTemp1[:-2])
fts_object.data[2][0] = sitk.GetImageFromArray(npTemp2[:-2])
fts_object.data[4][0] = sitk.GetImageFromArray(npTemp3[:-2])

fts_object.createkDimImages(3,'png')
fts_object.saveToHardDisk('.\\output_dir\\')
exit()'''

# Example for creating elastic deformed data
'''fts_object.load('.\\input\\')
fts_object.load('.\\ground\\')
fts_object.elasticDeformation(512,512,5,10)
fts_object.saveToHardDisk('.\\input\\')
fts_object.saveToHardDisk('.\\ground\\',1)
exit()'''

# Example of converting dicom files to nifti
'''fts_object.loadDICOMData('.\\SER_0015\\')
fts_object.loadDICOMData('.\\SER_0016\\')
fts_object.saveToHardDisk('.\\top\\',0)
fts_object.saveToHardDisk('.\\bot\\',1)
exit()'''

# Example of a preprocessing pipeline for CT-Data
'''toploaddir = '.\\top\\'
botloaddir = '.\\bot\\'
outputdir = '.\\output_dir\\'

fts_object.setParameters(False, False,-600,10,-1000)
fts_object.load(toploaddir) # Folder contains only top data
fts_object.load(botloaddir) # Folder contains only bottom data
#fts_object.load('.\\IMediaExport 01\\')
fts_object.mergeDataRAW(toploaddir,botloaddir)
fts_object.saveToHardDisk(outputdir+'merge\\')
fts_object.saveToHardDisk(outputdir+'merge\\',1)
fts_object.clearData()
fts_object.load(outputdir+'merge\\')
fts_object.load(outputdir+'merge\\')
fts_object.concToArray('dump.nii.gz')
fts_object.saveToHardDisk(outputdir)
exit()'''

# Example of a preprocessing pipeline for Segmentation data
'''toploaddir = '\\top\\'
botloaddir = '.\\bot\\'
outputdir = '.\\output_dir\\'

fts_object.setParameters(False, True,-600,10,-1000)
fts_object.load(toploaddir) # Folder contains only top data
fts_object.load(botloaddir) # Folder contains only bottom data
#fts_object.load('.\\IMediaExport 01\\')
fts_object.mergeDataRAW(toploaddir,botloaddir)
fts_object.saveToHardDisk(outputdir+'temp\\')
fts_object.saveToHardDisk(outputdir+'merge\\',1)
fts_object.clearData()
fts_object.load(outputdir+'temp\\_middle_top.nii.gz')
fts_object.load(outputdir+'temp\\_middle_bot.nii.gz')
os.remove(outputdir+'temp\\_middle_top.nii.gz')
os.remove(outputdir+'temp\\_middle_bot.nii.gz')
fts_object.mergeData()
fts_object.data[1][0] = 'patient19.middle.nii'
fts_object.saveToHardDisk(outputdir+'merge\\')
fts_object.clearData()
fts_object.load(outputdir+'merge\\')
fts_object.load(outputdir+'merge\\')
fts_object.concToArray('dump.nii.gz)
fts_object.saveToHardDisk(outputdir)
exit()'''

print('End!')
