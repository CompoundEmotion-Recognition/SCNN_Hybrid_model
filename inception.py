#importing python modules
import os
import ktrain
from ktrain import vision as vis
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
from sklearn.metrics import classification_report, confusion_matrix

#image reshaping
train = ImageDataGenerator(rescale=1/255)
validation = ImageDataGenerator(rescale= 1./255)

#defining the Train and test data 
train_data = train.flow_from_directory('/content/drive/MyDrive/dataset/train',
                                          target_size=(200,200),
                                          batch_size=3,
                                          class_mode='categorical')

test_data = validation.flow_from_directory('/content/drive/MyDrive/dataset/test',
                                          target_size=(200,200),
                                          batch_size=3,
                                          class_mode='categorical')


#Using Transfer learning on Inception model
model= vis.image_regression_model('pretrained_inception',
                                  train_data=train_data,
                                  val_data=test_data)
model.summary()

#Defining the metrics scores
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
#Compiling the Inception model
model.compile(loss= 'categorical_crossentropy',
              optimizer = RMSprop(lr=0.001),
              metrics=['Precision','accuracy','Recall','AUC',f1_score])


#Defining the confusion matrix for the model
Y_pred = model.predict(train_data )
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(train_data.classes, y_pred))
print('Classification Report')
target_names = ['0', '1','2','3','4','5','6','7','8','9','10']
print(classification_report(train_data.classes, y_pred, target_names=target_names))


#Traing the model with 20 epochs
learner = ktrain.get_learner(model=model,
                             train_data=train_data,
                             val_data=test_data,
                             batch_size=64)
learner.fit_onecycle(1e-4,20)


#Plot of loss
plt.plot(learner.history.history['loss'], label='train loss')
plt.plot(learner.history.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')
# plot the accuracy
plt.plot(learner.history.history['accuracy'], label='train acc')
plt.plot(learner.history.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')
#plot the precision
plt.plot(learner.history.history['precision'], label='train prec')
plt.plot(learner.history.history['val_precision'], label='val prec')
plt.legend()
plt.show()
plt.savefig('AccVal_prec')
#plot recall
plt.plot(learner.history.history['recall'], label='train recall')
plt.plot(learner.history.history['val_recall'], label='val recall')
plt.legend()
plt.show()
plt.savefig('AccVal_rec')
#plot f1_score
plt.plot(learner.history.history['f1_score'], label='train f1')
plt.plot(learner.history.history['val_f1_score'], label='val f1')
plt.legend()
plt.show()
plt.savefig('AccVal_f1')


