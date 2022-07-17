import tensorflow as tf
import os
from keras.utils import image_dataset_from_directory,image_utils
import variables
from modelStructure import LeNet
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

listExtension = ['.bmp','.jpg','.png','.jpeg','.BMP','.PNG','.JPG','.JPEG']
trainFolder = os.path.join(variables.dataDir,variables.trainFolder)
className = sorted(os.listdir(trainFolder))

"""Load tensor data"""
#Load single tensor
@staticmethod
def map_images_and_labels(filePath):
    try:
        """Load images"""
        img = tf.io.read_file(filePath)
        img = tf.io.decode_png(img,channels=variables.channel)
        img = tf.image.resize(img,size=(variables.input_shape[0],variables.input_shape[1]))
        """Standalize"""
        img = img/255.0

        """Load label"""
        label = tf.strings.split(filePath,os.path.sep)[-2]
        oneHot = label == className
        encodedLabel = tf.argmax(oneHot)
        return img,encodedLabel
    except:
        pass

#Load a list of tensor
def loadTensorData(folderPath = trainFolder,shuffle = True,repeat = 1,batch_size = variables.batch_size):
    if os.path.exists(folderPath):
        """Dataset Processing"""
        dataset = tf.data.Dataset.list_files(file_pattern=f"{folderPath}/*/*",shuffle=shuffle)
        #Repeat Data
        dataset = dataset.repeat(count=repeat)
        #Map data
        dataset = dataset.map(map_images_and_labels,num_parallel_calls=tf.data.AUTOTUNE)
        #Augurmentation
        augurment = LeNet.augurment(shift_range=variables.shift_range, rotation_range=variables.rotation_range,
                                    zoom_range=variables.zoom_range)
        dataset = dataset.map(lambda x, y: (augurment(x), y),num_parallel_calls=tf.data.AUTOTUNE)
        #Batch data
        dataset = dataset.batch(batch_size=batch_size)
        #Prefetch data
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    else:
        raise Exception("Path is not existed!")

#Load a list of tensor
def loadTensorFolder(folderPath = trainFolder,shuffle = True,repeat = 1,batch_size = variables.batch_size):
    if os.path.exists(folderPath):
        """Dataset Processing"""
        dataset = image_dataset_from_directory(directory=folderPath,image_size=(variables.input_shape[0],
                                            variables.input_shape[1]),batch_size=batch_size,shuffle=shuffle)
        #Repeat Data
        dataset = dataset.repeat(count=repeat)
        #standalize data
        dataset = dataset.map(lambda x,y: (x/255.0,y),num_parallel_calls=tf.data.AUTOTUNE)

        # Augurmentation
        augurment = LeNet.augurment(shift_range=variables.shift_range,rotation_range=variables.rotation_range
                                    ,zoom_range=variables.zoom_range)
        dataset = dataset.map(lambda x, y: (augurment(x), y),num_parallel_calls=tf.data.AUTOTUNE)
        #Prefetch data
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    else:
        raise Exception("Path is not existed!")

"""Load Data Generator data"""
def loadDataGenerator(folderPath = trainFolder,batch_size = variables.batch_size,augurment = False,rescale = True,shuffle = True):
    #Rescale
    if rescale == True:
        factor = 1/255.0
    else:
        factor = 1

    """Initialize data object"""
    if augurment:
        dataGen = ImageDataGenerator(rescale=factor,width_shift_range=variables.shift_range,height_shift_range=variables.shift_range,
                                 zoom_range=variables.zoom_range,rotation_range=int(360*variables.rotation_range))
    else:
        dataGen = ImageDataGenerator(rescale=factor)

    #Check folder path
    if os.path.exists(folderPath):
        data_flow = dataGen.flow_from_directory(directory=folderPath,target_size=(variables.input_shape[0],variables.input_shape[1]),
                                                shuffle=shuffle,batch_size=batch_size,class_mode="sparse")
        return data_flow
    else:
        raise Exception("Path is not existed!")

"""Get single/subsequent array image from folder"""
def getRamdomlyImage(folderPath,numberOfImages,target_size = (224,224),standalize = False):
    if os.path.exists(folderPath):
        """Get all the file"""
        listFile = sorted(os.listdir(folderPath))
        """Array of filename"""
        listPath = []
        for fileName in listFile:
            """Check whether file is image or not"""
            if os.path.splitext(fileName)[1] in listExtension:
                """Concat the path"""
                filePath = os.path.join(folderPath,fileName)
                """Append to list"""
                listPath.append(filePath)

        listPath = np.array(listPath)
        """Check total image"""
        if len(listPath) == 0:
            raise Exception("Image is empty")
        elif numberOfImages <= len(listPath):
            """Case for true statement"""
            """Shuffle and randomly pick from array"""
            shuffleList = np.random.permutation(listPath)
            choiceList = np.random.choice(a=shuffleList,size=numberOfImages,replace=False)
            arrayImages = []
            for pathImage in choiceList:
                img = image_utils.load_img(pathImage,target_size=target_size)
                img = image_utils.img_to_array(img)
                """If standalize """
                if standalize:
                    img = img/255.0
                """Add dimension"""
                img = np.expand_dims(img,0)
                arrayImages.append(img)
            arrayImages = np.vstack(arrayImages)
            return arrayImages
        else:
            print("Number choice is over total images")

    else:
        raise Exception("Folder is not correct")

