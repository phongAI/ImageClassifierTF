from keras import models
import os
import matplotlib.pyplot as plt
import preprocess
import variables
import numpy as np
model = models.load_model(os.path.join(variables.SaveWeightFolder,"MobileNet.h5"))

def main():
    #Image path
    valPath = os.path.join(variables.dataDir,variables.valFolder)
    print(sorted(os.listdir(os.path.join(variables.dataDir,variables.trainFolder))))
    # """Predict list of image"""
    listImages = preprocess.getRamdomlyImage(folderPath=valPath,numberOfImages=5,standalize=True)
    prediction = model.predict(listImages)
    print(np.argmax(prediction,axis=1))
    for i in range(len(listImages)):
        plot = plt.subplot(1,len(listImages),i+1)
        plt.imshow(listImages[i])
    plt.show()

if __name__ == "__main__":
    main()