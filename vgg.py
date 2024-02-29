#import libraries
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as numpy
import matplotlib.pyplot as plt
from keras import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
import keras.backend as K
from tensorflow.keras.optimizers import RMSprop
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from datetime import datetime
from keras.callbacks import ModelCheckpoint
import warnings
warnings.filterwarnings("ignore",category=FutureWarning)
IMAGE_SIZE = [224,224]
#Load and define the datasets
train_path='/content/drive/MyDrive/dataset/train'
validation_path='/content/drive/MyDrive/dataset/test'
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')
test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')
train_set= train_datagen.flow_from_directory(train_path,
                                             target_size=(224,224),
                                             batch_size=3,
                                             class_mode='categorical')
test_set= test_datagen.flow_from_directory(validation_path,
                                           target_size=(224,224),
                                           batch_size=3,
                                           class_mode='categorical')

train_set.class_indices


#define the metric values
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
#Compile the model
model.compile(loss= 'categorical_crossentropy',
              optimizer = RMSprop(lr=0.001),
              metrics=['Precision','accuracy','Recall','AUC',f1_score])

#calculating the confusion matrix
Y_pred = model.predict_generator(train_set )
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(train_set.classes, y_pred))
print('Classification Report')
target_names = ['0', '1','2','3','4','5','6','7','8','9','10']
print(classification_report(train_set.classes, y_pred, target_names=target_names))

#Run and train the model for 20 epochs
checkpoint = ModelCheckpoint(filepath= 'mymodel.h5',
                             verbose=2, save_best_only=True)
callbacks = [checkpoint]
start = datetime.now()
model_history=model.fit_generator(
    train_set,
    validation_data=test_set,
    epochs=20

)
duration = datetime.now() - start
print("training completed in time", duration)

#plot the loss
plt.plot(model_history.history['loss'], label='train loss')
plt.plot(model_history.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# plot the accuracy
plt.plot(model_history.history['accuracy'], label='train acc')
plt.plot(model_history.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

#plot the precision
plt.plot(model_history.history['precision'], label='train prec')
plt.plot(model_history.history['val_precision'], label='val prec')
plt.legend()
plt.show()
plt.savefig('AccVal_prec')

#plot recall
plt.plot(model_history.history['recall'], label='train recall')
plt.plot(model_history.history['val_recall'], label='val recall')
plt.legend()
plt.show()
plt.savefig('AccVal_rec')

#plot f1_score
plt.plot(model_history.history['f1_score'], label='train f1')
plt.plot(model_history.history['val_f1_score'], label='val f1')
plt.legend()
plt.show()
plt.savefig('AccVal_f1')
