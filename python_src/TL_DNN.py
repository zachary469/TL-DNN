# This program imports a pretrained deep neural network from Keras
# and applies transfer learning to train it on the provided dataset:
import pandas as pd
import numpy as np
import os
import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam
import pickle
'''
SETUP SECTION: Choose one of three pretrained models
with the settings below. Make sure to use matching
choices for KerasDNN, preprocess_input, and base_layer_count.
To use a different pretrained network, add your own
option here, and also the corresponding plotting and
export lines at the end of this file.
'''
# Select one from the three below:
from keras.applications import MobileNet as KerasDNN
#from keras.applications import DenseNet121 as KerasDNN
#from keras.applications import Xception as KerasDNN
# Select one from the three below in agreement with the above choice:
from keras.applications.mobilenet import preprocess_input
#from keras.applications.densenet import preprocess_input
#from keras.applications.xception import preprocess_input
# Select one from the 3 below, corresponding to the choice of base model:
base_layer_count=88 # MobileNet
#base_layer_count=121 # DenseNet
#base_layer_count=126 # Xception
'''
End of SETUP SECTION
'''
# We discard the final layer:
base_model=KerasDNN(weights='imagenet',include_top=False)
# We add a few extra dense layers and finally a 37-node layer
# to perform the classification.
# We make 4 different networks with 2-5 layers:
xx=[None]*4
preds=[None]*4
model=[None]*4
for ii in range(4):
    xx[ii]=base_model.output
    xx[ii]=GlobalAveragePooling2D()(xx[ii])
xx[1]=Dense(1024,activation='relu')(xx[ii])
xx[2]=Dense(1024,activation='relu')(xx[ii])
xx[2]=Dense(1024,activation='relu')(xx[ii])
xx[3]=Dense(1024,activation='relu')(xx[ii])
xx[3]=Dense(1024,activation='relu')(xx[ii])
xx[3]=Dense(1024,activation='relu')(xx[ii])
for ii in range(4):
    xx[ii]=Dense(512,activation='relu')(xx[ii])
    preds[ii]=Dense(37,activation='softmax')(xx[ii])

# Here we specify the input for the model
for ii in range(4):
    model[ii]=Model(inputs=base_model.input,outputs=preds[ii])
# For debugging, here we can print the architecture of the model to standard output:
#    print('\n The network will have the following layers:')
#    for jj,layer in enumerate(model.layers):
#        print(jj,layer.name)
# Here we specify that all pre-trained weights should remain fixed, i.e. only the newly added
# layers will be trained:
    for layer in model[ii].layers:
            layer.trainable=False
    for layer in model[ii].layers[base_layer_count:]:
        layer.trainable=True

# Here we process the input images and reserve 25% as a validation set:
print('\n Importing Image Data')
train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input,validation_split=0.25)
train_generator=train_datagen.flow_from_directory('../images',target_size=(224,224),
    color_mode='rgb',batch_size=32,class_mode='categorical',shuffle=True,subset='training')
val_generator=train_datagen.flow_from_directory('../images',target_size=(224,224),
    color_mode='rgb',batch_size=32,class_mode='categorical',shuffle=True,subset='validation')

print('\nImage Data Imported')
# Here we set the steps per epoch:
step_size_train=train_generator.n//train_generator.batch_size
step_size_val=val_generator.n//val_generator.batch_size
# Here we set the number of epochs over which to train the network.
# Tests show that 20 should suffice for convergence:
numepochs=20

# Here we loop over the 4 networks defined above:

for imodel in range(4):
    print('\nCompiling neural network',imodel)
# Here we compile the model using the Adam optimizer and categorical cross entropy loss function:
    model[imodel].compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
    print('\nTraining neural network',imodel)
# Here we train the network:
    trained_model = model[imodel].fit_generator(generator=train_generator,validation_data=val_generator,
        steps_per_epoch=step_size_train,epochs=numepochs,validation_steps=step_size_val)
    print('\nTraining complete')
# Here we plot the performance metrics:
    NN = np.arange(0, numepochs)
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(NN, trained_model.history['loss'], label='Training Loss')
    plt.plot(NN, trained_model.history['val_loss'], label='Validation Loss')
    plt.plot(NN, trained_model.history['acc'], label='Training Accuracy')
    plt.plot(NN, trained_model.history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Loss/Accuracy')
    plt.legend()
    if imodel==0:
        if base_layer_count==88:
            plt.title('Training Loss and Accuracy (TL MobileNet 2L)')
            plt.savefig('figures/TL_MobileNet_2L')
        if base_layer_count==121:
            plt.title('Training Loss and Accuracy (TL DenseNet 2L)')
            plt.savefig('figures/TL_DenseNet_2L')
        if base_layer_count==126:
            plt.title('Training Loss and Accuracy (TL Xception 2L)')
            plt.savefig('figures/TL_Xception_2L')
# Here we export the neural network for future use:
        print('\nExporting neural network',imodel)
        if base_layer_count==88:
            model[imodel].save('models/TL_MobileNet_2L')
        if base_layer_count==121:
            model[imodel].save('models/TL_DenseNet_2L')
        if base_layer_count==126:
            model[imodel].save('models/TL_Xception_2L')
    if imodel==1:
        if base_layer_count==88:
            plt.title('Training Loss and Accuracy (TL MobileNet 3L)')
            plt.savefig('figures/TL_MobileNet_3L')
        if base_layer_count==121:
            plt.title('Training Loss and Accuracy (TL DenseNet 3L)')
            plt.savefig('figures/TL_DenseNet_3L')
        if base_layer_count==126:
            plt.title('Training Loss and Accuracy (TL Xception 3L)')
            plt.savefig('figures/TL_Xception_3L')
# Here we export the neural network for future use:
        print('\nExporting neural network',imodel)
        if base_layer_count==88:
            model[imodel].save('models/TL_MobileNet_3L')
        if base_layer_count==121:
            model[imodel].save('models/TL_DenseNet_3L')
        if base_layer_count==126:
            model[imodel].save('models/TL_Xception_3L')
    if imodel==2:
        if base_layer_count==88:
            plt.title('Training Loss and Accuracy (TL MobileNet 4L)')
            plt.savefig('figures/TL_MobileNet_4L')
        if base_layer_count==121:
            plt.title('Training Loss and Accuracy (TL DenseNet 4L)')
            plt.savefig('figures/TL_DenseNet_4L')
        if base_layer_count==126:
            plt.title('Training Loss and Accuracy (TL Xception 4L)')
            plt.savefig('figures/TL_Xception_4L')
# Here we export the neural network for future use:
        print('\nExporting neural network',imodel)
        if base_layer_count==88:
            model[imodel].save('models/TL_MobileNet_4L')
        if base_layer_count==121:
            model[imodel].save('models/TL_DenseNet_4L')
        if base_layer_count==126:
            model[imodel].save('models/TL_Xception_4L')
    if imodel==3:
        if base_layer_count==88:
            plt.title('Training Loss and Accuracy (TL MobileNet 5L)')
            plt.savefig('figures/TL_MobileNet_5L')
        if base_layer_count==121:
            plt.title('Training Loss and Accuracy (TL DenseNet 5L)')
            plt.savefig('figures/TL_DenseNet_5L')
        if base_layer_count==126:
            plt.title('Training Loss and Accuracy (TL Xception 5L)')
            plt.savefig('figures/TL_Xception_5L')
# Here we export the neural network for future use:
        print('\nExporting neural network',imodel)
        if base_layer_count==88:
            model[imodel].save('models/TL_MobileNet_5L')
        if base_layer_count==121:
            model[imodel].save('models/TL_DenseNet_5L')
        if base_layer_count==126:
            model[imodel].save('models/TL_Xception_5L')









