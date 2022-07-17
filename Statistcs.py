import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing import image
from tensorflow import keras
import numpy as np
import os
import seaborn as sb
from sklearn.metrics import confusion_matrix,classification_report
onehotEncoder = LabelEncoder()
class infoStatistics():
    @staticmethod
    def directoryInfo(path,showBarFigure = False):
        if os.path.exists(path):
            itemClasses = []
            classesName = os.listdir(path)
            #Count class
            totalClasses = len(classesName)
            if(totalClasses == 0):
                raise Exception("Dataset is empty!")
            else:
                for (i,smallClass) in enumerate(classesName):
                    folderPath = os.path.join(path,smallClass)
                    itemCount = len(os.listdir(folderPath))
                    itemClasses.append(itemCount)
        else:
            raise Exception("Dataset is not existed!")

        #Show bar figure
        if showBarFigure:
            figureStatistics.showBarFigure(classesName,itemClasses)
        return itemClasses,classesName

    @staticmethod
    def getInput(path,target_size = (224,224)):
        if os.path.exists(path):
            itemClasses = []
            classesName = os.listdir(path)
            # Count class
            totalClasses = len(classesName)
            if (totalClasses == 0):
                raise Exception("Dataset is empty!")
            else:
                imageArray = []
                for smallClass in classesName:
                    classPath = os.path.join(path,smallClass)
                    for imageName in os.listdir(classPath):
                        imagePath = os.path.join(classPath,imageName)
                        load_img = image.load_img(imagePath,target_size=target_size)
                        img_array = image.img_to_array(load_img)/255.0
                        imageArray.append([img_array])
        else:
            raise Exception("Dataset is not existed!")
        imageArray = np.vstack(imageArray)
        return imageArray

    @staticmethod
    def getLabel(itemClasses):
        labelArray = []
        if len(itemClasses) >0:
            for (i,num) in enumerate(itemClasses):
                labelArray.append(np.full(shape=(num,1),fill_value=i))
        else:
            raise Exception("Empty label!")
        label = np.vstack(labelArray)
        label = keras.utils.to_categorical(label)
        return label

    @staticmethod
    def LoadInfoFromDirectory(path,target_size = (224,224),label_mode = "binary"):
        if os.path.exists(path):
            datasets = image.image_dataset_from_directory(path,image_size=target_size,label_mode=label_mode)
            return datasets

class figureStatistics():
    @staticmethod
    def showBarFigure(x,y):
        recreatedArray = []
        for item in x:
            if len(item) >4:
                newItem = item[:4]
                recreatedArray.append(newItem)
            else:
                recreatedArray.append(newItem)
        x  = recreatedArray
        plt.bar(x,y)
        plt.title("Amount of images per class")
        plt.xlabel("Class")
        plt.ylabel("Total images")
        plt.autoscale()
        plt.show()

    @staticmethod
    def showImage(image):
        image = np.array(image)
        if np.ndim(image) == 3:
            plt.imshow(image)
            plt.show()
        else:
            raise Exception("Format incorrect!")

    @staticmethod
    def showConfusionMatrix(yPredict,yTrue):
        if np.ndim(yPredict) == np.ndim(yTrue):
            cm = confusion_matrix(yTrue,yPredict)
            sb.heatmap(cm,annot=True,fmt='g')
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted Class")
            plt.ylabel("Actual Class")
            plt.show()

    @staticmethod
    def showClassificationReport(yPredict,yTrue):
        print(classification_report(yTrue,yPredict))

    @staticmethod
    def showLossAndAccuracy(loss,acc,val_loss = None,val_acc = None):
        epoch = np.arange(len(loss))
        step = np.arange(start=0,stop=len(loss)+1,step=5)
        #Show train loss and train accuracy


        #Show all
        if val_loss == None:
            plt.plot(epoch, loss, label="Loss")
            plt.plot(epoch, acc, label="Accuracy")
            plt.autoscale()
            plt.xlabel("Total epochs:")
            plt.ylabel("Losses and Accuracies:")
            plt.title("Loss and accuracy overall")
            plt.legend()
            plt.show()
        else:
            #First figure (Loss)
            plot1 = plt.figure(1)
            plt.plot(epoch,loss, label="Loss")
            plt.plot(epoch,val_loss, label="Validation Loss")
            plt.xlabel("Total epoch:")
            plt.ylabel("Total losses:")
            plt.xlim([0, len(loss) + 1])
            if len(loss) > 10:
                plt.xticks(step)
            plt.legend()
            plt.title("Dependences between losses and epoches")

            #Second figure (Accuracy)
            plot2 = plt.figure(2)
            plt.plot(epoch, acc, label="Accuracy")
            plt.plot(epoch,val_acc, label="Validation Accuracy")
            plt.ylim([0,1.1])
            plt.axline(xy1=(0,0.90),xy2=(len(loss),0.90),c = "g",label = "Threshold: 90%")
            plt.xlabel("Total epoch:")
            plt.ylabel("Total accuracy:")
            if len(loss) > 10:
                plt.xticks(step)
            plt.xlim([0,len(loss)+1])
            plt.legend()
            plt.title("Dependences between accuracy and epoches")
            plt.show()
        #Show
