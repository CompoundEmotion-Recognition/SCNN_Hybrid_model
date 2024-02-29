# importing libraries
import numpy as np          
import pandas as pd                         
import matplotlib.pyplot as plt              
from keras.layers import Flatten, Dense                                          
from keras.models import Model                                                           
from keras.preprocessing.image import ImageDataGenerator , img_to_array, load_img             
from keras.applications.mobilenet import MobileNet, preprocess_input                  
from keras.losses import categorical_crossentropy                                  
import keras.backend as K
from keras.callbacks import Callback,ModelCheckpoint
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.callbacks import ModelCheckpoint, EarlyStopping

train_datagen = ImageDataGenerator(
     zoom_range = 0.2,
     shear_range = 0.2,
     horizontal_flip=True,                     #slants the shape of image
     rescale = 1./255
)
val_datagen = ImageDataGenerator(
     zoom_range = 0.2,
     shear_range = 0.2,
     horizontal_flip=True,                     #slants the shape of image
     rescale = 1./255
)
train_data = train_datagen.flow_from_directory(directory= "/content/drive/MyDrive/dataset/train",
                                               target_size=(224,224),
                                               batch_size=20,                                      #divide dataset into Number of Batches
                                  )
val_data = val_datagen.flow_from_directory(directory= "/content/drive/MyDrive/dataset/test",
                                               target_size=(224,224),
                                               batch_size=20,                                      #divide dataset into Number of Batches
                                  )
train_data.class_indices

# Working with pre trained model
base_model =  MobileNet( input_shape=(224,224,3), include_top= False )        #  include_top = false is allowing new o/p layer to be added and trained
for layer in base_model.layers:
  layer.trainable = False                      # moves all layer's weights from trainable to non-trainable.freezing the layer, indicating that this layer should not be trained.
z = Flatten()(base_model.output)
x = Dense(units=11 , activation='softmax' )(z)           # unit is o/p dimension , softmax-vector of values to a probability distribution(fr last layer) [p(happily sad)]
# creating our model.
model = Model(base_model.input, x)
model.summary()
model.compile(optimizer='adam', loss= categorical_crossentropy , metrics=['accuracy','AUC','Recall','Precision',f1_score])

## having early stopping and model check point
# A callback is an object that can perform actions at various stages of training (e.g. at the start or end of an epoch, before or after a single batch, etc).
# early stopping
es = EarlyStopping(monitor='val_accuracy',
                   min_delta= 0.0001 ,                   # an absolute change of less than min_delta, will count as no improvement.
                   patience= 7,                # Number of epochs with no improvement after which training will be stopped.
                   verbose= 1,                     # Verbosity mode, 0 or 1. Mode 0 is silent, and mode 1 displays messages when the callback takes an action.
                   mode='max')               # decision to overwrite the current save file is made based on either the maximization or the minimization of the monitored quantity.
# model check point
mc = ModelCheckpoint(filepath="/content/drive/MyDrive/dataset/best_model.h5",
                     monitor= 'val_accuracy',
                     verbose= 1,
                     save_best_only= True,
                     mode = 'max')
# puting call back in a list
#callbacks = [ModelCheckpoint("/content/drive/MyDrive/dataset/best_model.h5", save_best_only=True)]
call_back = [es, mc]
hist = model.fit(train_data,

                           epochs= 20,
                           validation_data= val_data,
                           validation_steps= 1,
                           callbacks=[es,mc])
#Confusion matrix
Y_pred = model.predict(val_data, 809)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix\n')
print(confusion_matrix(val_data.classes, y_pred))
print('\nClassification Report')
target_names = ['0', '1','2','3','4','5','6','7','8','9','10']
print(classification_report(val_data.classes, y_pred,target_names=target_names ))

#plot accuracy
plt.plot(h['accuracy'])
plt.plot(h['val_accuracy'],c = "red")
plt.title("acc vs v-acc")
plt.show()
#plot loss
plt.plot(h['loss'])
plt.plot(h['val_loss'] , c = "red")
plt.title("loss vs v-loss")
plt.show()
#plot AUC
plt.plot(h['auc'])
plt.plot(h['val_auc'] , c = "red")
plt.title("auc vs val_auc")
plt.show()
#Plot recall
plt.plot(h['recall'])
plt.plot(h['val_recall'] , c = "red")
plt.title("recall vs val_recall")
plt.show()
#plot precesion
plt.plot(h['precision'])
plt.plot(h['val_precision'] , c = "red")
plt.title("precision vs val_precision")
plt.show()
#plot f1_score
plt.plot(h['f1_score'])
plt.plot(h['val_f1_score'],c = "red")
plt.title("f1 score vs v-f1 score")
plt.show()



