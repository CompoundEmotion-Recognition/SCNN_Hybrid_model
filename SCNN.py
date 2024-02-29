#importing libraries
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import callbacks,optimizers
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
import keras.backend as K
# Loading Dataset
train = ImageDataGenerator(rescale=1/255)
validation = ImageDataGenerator(rescale= 1./255)
train_dataset = train.flow_from_directory('/content/drive/MyDrive/dataset/train',
                                          target_size=(200,200),
                                          batch_size=3,
                                          class_mode='categorical')

validation_dataset = validation.flow_from_directory('/content/drive/MyDrive/dataset/test',
                                          target_size=(200,200),
                                          batch_size=3,
                                          class_mode='categorical')

train_dataset.class_indices
train_dataset.classes


#defining the SCNN Model
model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(200,200,3)),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    #
                                    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    #
                                     tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
                                    x=tf.keras.layers.MaxPool2D(2,2),
                                    #
                                    y = tf.keras.layers.Flatten(),
                                    ##
                                    tf.keras.layers.Dense(512,activation='relu'),
                                    ##
                                    tf.keras.layers.Dense(1,activation='softmax')
                                    ])
#Defining the metrics parameters
def f1_score(y_true, y_pred):    
    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        
        recall = TP / (Positives+K.epsilon())    
        return recall 
    
    
    def precision_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    
        precision = TP / (Pred_Positives+K.epsilon())
        return precision 
    
    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)
    
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
# Model Compilation
model.compile(loss= 'categorical_crossentropy',
              optimizer = RMSprop(lr=0.001),
              metrics=['Precision','accuracy','Recall','AUC',f1_score])

model.summary()
#Model training for 20 epochs
model_fit = model.fit(train_dataset,
                      validation_data = validation_dataset,
                      epochs=20,
                      steps_per_epoch=2)
#plot loss
plt.plot(model_fit.history['loss'], label='train loss')
plt.plot(model_fit.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# plot the accuracy
plt.plot(model_fit.history['accuracy'], label='train acc')
plt.plot(model_fit.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

#plot the precision
plt.plot(model_fit.history['Precision'], label='train prec')
plt.plot(model_fit.history['val_precision'], label='val prec')
plt.legend()
plt.show()
plt.savefig('AccVal_prec')

#plot recall
plt.plot(model_fit.history['Recall'], label='train recall')
plt.plot(model_fit.history['val_Recall'], label='val recall')
plt.legend()
plt.show()
plt.savefig('AccVal_rec')

#plot f1_score
plt.plot(model_fit.history['f1_score'], label='train f1')
plt.plot(model_fit.history['val_f1_score'], label='val f1')
plt.legend()
plt.show()
plt.savefig('AccVal_f1')

