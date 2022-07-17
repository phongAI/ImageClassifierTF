import os
from keras import models,layers,optimizers,callbacks
from keras import applications
import variables

#Build-up class
class LeNet():
    #Init
    def __init__(self,input_shape = variables.input_shape,classes = variables.classes):
        #Get input parameter
        self.input_shape = input_shape
        self.classes = classes

        # Return optimizer
        self.optimizer = self.AdamOptimizer()

    #Optimizer
    def AdamOptimizer(self,learning_rate = variables.learning_Rate,beta1 = 0.9,beta2 = 0.999,epsilon = 1e-7,amsgrad = False):
        optimizer = optimizers.Adam(learning_rate=learning_rate,beta_1=beta1,beta_2=beta2,epsilon=epsilon,amsgrad=amsgrad)
        self.optimizer = optimizer
        # Rebuild model
        self.model = self.build()

    def SGDOptimizer(self,learning_rate = variables.learning_Rate,momentum = 0.9,nesterov = False,amsgrad = False):
        optimizer = optimizers.sgd_experimental.SGD(learning_rate=learning_rate,momentum=momentum,nesterov=nesterov,amsgrad=amsgrad)
        self.optimizer = optimizer
        # Rebuild model
        self.model = self.build()

    #Data augurmentation
    @staticmethod
    def augurment(shift_range = 0.1,rotation_range = 0.1,zoom_range = 0.1):
        augurment = models.Sequential([
            layers.RandomTranslation(height_factor=shift_range,width_factor=shift_range),
            layers.RandomRotation(factor=rotation_range),
            layers.RandomZoom(height_factor=zoom_range),
            layers.RandomFlip(mode="vertical")
        ])
        return augurment
        
    #build model
    def build(self):
        model = models.Sequential([
            layers.Conv2D(16, 3, padding="same", input_shape=self.input_shape, activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Conv2D(16, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.25),
            layers.Dense(self.classes, activation="softmax")
        ])
        model.compile(optimizer=self.optimizer,loss="sparse_categorical_crossentropy",metrics=['accuracy'])
        return model

    def summary(self):
        return self.model.summary()

    def fit(self,data_train,epoch = variables.epoch):
        #Save file path
        save_path = os.path.join(variables.SaveWeightFolder,f"{self.__class__.__name__}.h5")
        #Call-back function
        modeCheck = callbacks.ModelCheckpoint(filepath=save_path,monitor="loss",save_best_only=True)
        reduceLr = callbacks.ReduceLROnPlateau(monitor='loss',patience=5)
        earlyStop = callbacks.EarlyStopping(monitor='loss',patience=5)

        #Fit method  			
        self.model.fit(x=data_train,epochs=epoch,steps_per_epoch=len(data_train),callbacks=[modeCheck,reduceLr,earlyStop])

#Transfer Learning
class MobileNet(LeNet):
    def __init__(self,input_shape = variables.input_shape,classes = variables.classes):
        super().__init__(input_shape,classes)
        
    def build(self):
        rootModel = applications.MobileNet(input_shape=self.input_shape,include_top=False,classes=self.classes)
        input = rootModel.input
        output1 = rootModel.output
        x = layers.Flatten()(output1)
        x = layers.Dense(256,activation="relu")(x)
        x = layers.Dropout(0.25)(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.25)(x)
        output2 = layers.Dense(self.classes,activation="softmax")(x)
        model = models.Model(input,output2)
        model.compile(optimizer=self.optimizer, loss="sparse_categorical_crossentropy", metrics=['accuracy'])
        return model

class VGG16(LeNet):
    def __init__(self,input_shape = variables.input_shape,classes = variables.classes):
        super().__init__(input_shape,classes)

    def build(self):
        rootModel = applications.VGG16(input_shape=self.input_shape,include_top=False,classes=self.classes)
        input = rootModel.input
        output1 = rootModel.output
        x = layers.Flatten()(output1)
        x = layers.Dense(256,activation="relu")(x)
        x = layers.Dropout(0.25)(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.25)(x)
        output2 = layers.Dense(self.classes,activation="softmax")(x)
        model = models.Model(input,output2)
        model.compile(optimizer=self.optimizer, loss="sparse_categorical_crossentropy", metrics=['accuracy'])
        return model
