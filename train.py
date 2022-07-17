import os
import variables
from modelStructure import LeNet,MobileNet,VGG16
import preprocess

def main():
    """Create folder if neccessay"""
    #Save weight folder
    os.makedirs(variables.SaveWeightFolder,exist_ok=True)
    #Train data folder
    os.makedirs(variables.dataDir,exist_ok=True)

    """Load data"""
    dataset = preprocess.loadTensorData()
    #dataset = preprocess.loadTensorFolder()
    #dataset = preprocess.loadDataGenerator(augurment=True)

    #MobileNet
    mobileNet = MobileNet()
    mobileNet.fit(data_train=dataset,epoch=variables.epoch)


if __name__ == '__main__':
    main()

