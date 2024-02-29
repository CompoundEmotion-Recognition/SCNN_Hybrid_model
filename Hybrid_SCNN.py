#Importing libraries
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import callbacks,optimizers
import keras
from tensorflow.keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
import keras.backend as K
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
#Load the dataset
train = ImageDataGenerator(rescale=1/255)
validation = ImageDataGenerator(rescale= 1./255)

train_dataset = train.flow_from_directory('/content/drive/MyDrive/dataset/train',
                                          target_size=(200,200),
                                          batch_size=32,
                                          class_mode='categorical')

validation_dataset = validation.flow_from_directory('/content/drive/MyDrive/dataset/test',
                                          target_size=(200,200),
                                          batch_size=32,
                                          class_mode='categorical')

df = pd.read_csv('/content/drive/MyDrive/anjHOG.csv', header=None)
df.head()
#Define the HYBRID_S-CNN model
visible = Input(shape=(200,200,3))
conv1 = Conv2D(16,(3,3), input_shape=(200,200,3), activation='relu')(visible)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = Conv2D(32,(3,3), activation='relu')(pool1)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
conv3 = Conv2D(64,(3,3), activation='relu')(pool2)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
flat = Flatten()(pool3)
concat_ = keras.layers.Concatenate()([flat, df])
hidden1 = Dense(512, activation='relu')(flat)
output = Dense(11, activation='softmax')(hidden1)
model = Model(inputs=visible, outputs=output)
# summarize layers
print(model.summary())
# plot graph
plot_model(model, to_file='convolutional_neural_network.png')
#Define the metrics values
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
              optimizer = RMSprop(learning_rate=0.001),
              metrics=['Precision','accuracy','Recall','AUC',f1_score])
#Confusion matrics evaluation
Y_pred = model.predict(train_dataset )
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(train_dataset.classes, y_pred))
print('Classification Report')
target_names = ['0', '1','2','3','4','5','6','7','8','9','10']
print(classification_report(train_dataset.classes, y_pred, target_names=target_names))

#Train the model with 20 epochs
h = model.fit(train_dataset, epochs=100, validation_data=validation_dataset)
#Plot loss
plt.plot(h.history['loss'], label='train loss')
plt.plot(h.history['val_loss'], label='val loss')
plt.legend()
plt.savefig('LossVal_loss')
plt.xlabel('epochs-->')
plt.ylabel('loss-->')
plt.show()

#plot the precision
plt.plot(h.history['precision'], label='train prec')
plt.plot(h.history['val_precision'], label='val prec')
plt.legend()
plt.xlabel('epochs-->')
plt.ylabel('precision-->')
plt.show()
plt.savefig('AccVal_prec')

#plot recall
plt.plot(h.history['recall'], label='train recall')
plt.plot(h.history['val_recall'], label='val recall')
plt.legend()
plt.xlabel('epochs-->')
plt.ylabel('recall-->')
plt.show()
plt.savefig('AccVal_rec')

#plot f1_score
plt.plot(h.history['f1_score'], label='train f1')
plt.plot(h.history['val_f1_score'], label='val f1')
plt.legend()
plt.xlabel('epochs-->')
plt.ylabel('f1_score-->')
plt.show()
plt.savefig('AccVal_f1')

# plot the accuracy
plt.plot(h.history['accuracy'], label='train acc')
plt.plot(h.history['val_accuracy'], label='val acc')
plt.legend()
plt.xlabel('epochs-->')
plt.ylabel('accuracy-->')
plt.show()
plt.savefig('AccVal_acc')
#plot the AUC
plt.plot(h.history['auc'], label='train AUC')
plt.plot(h.history['val_auc'], label='val AUC')
plt.legend()
plt.xlabel('epochs-->')
plt.ylabel('auc-->')
plt.show()
plt.savefig('AUCVal_AUC')

