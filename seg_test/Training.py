import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

sns.set(style="whitegrid")
import os
import glob as gb
import cv2
import tensorflow as tf
import tensorflow.keras

from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from sklearn.metrics import confusion_matrix, classification_report
from random import randint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

trainpath = 'seg_data/seg_train/'
testpath = 'seg_data/seg_test/'
predpath = 'seg_data/seg_pred/'

print("train_data:")
for folder in os.listdir(trainpath + 'seg_train'):
    files = gb.glob(pathname=str(trainpath + 'seg_train//' + folder + '/*.jpg'))
    print(f'For training data , found {len(files)} in folder {folder}')

print("test_data:")
for folder in os.listdir(testpath + 'seg_test'):
    files = gb.glob(pathname=str(testpath + 'seg_test//' + folder + '/*.jpg'))
    print(f'For testing data , found {len(files)} in folder {folder}')

print('pred_data:')
files = gb.glob(pathname=str(predpath + 'seg_pred/*.jpg'))
print(f'For Prediction data , found {len(files)}')

code = {'buildings': 0, 'forest': 1, 'glacier': 2, 'mountain': 3, 'sea': 4, 'street': 5}


def getcode(n):  # A function to get the code back
    for x, y in code.items():
        if n == y:
            return x



train_path = 'seg_data/seg_train/seg_train/'
validation_path = 'seg_data/seg_test/seg_test/'

train_datagen = ImageDataGenerator(rescale=1.0 / 255,
                                   zoom_range=0.2,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   fill_mode='nearest')
train_generator = train_datagen.flow_from_directory(train_path,
                                                    target_size=(64, 64),
                                                    batch_size=50,
                                                    shuffle=True)

validation_datagen = ImageDataGenerator(rescale=1.0 / 255)
validation_generator = validation_datagen.flow_from_directory(validation_path,
                                                              target_size=(64, 64),
                                                              batch_size=50,
                                                              shuffle=True)

labels = {value: key for key, value in train_generator.class_indices.items()}
print(labels)
for key, value in labels.items():
    print(f"{key} : {value}")


def create_model():
    model = Sequential([
        Conv2D(filters=16, kernel_size=(7, 7), padding='valid', input_shape=(64, 64, 3)),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),

        Conv2D(filters=32, kernel_size=(7, 7), padding='valid', kernel_regularizer=l2(0.00005)),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),

        Conv2D(filters=64, kernel_size=(7, 7), padding='valid', kernel_regularizer=l2(0.00005)),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),

        Flatten(),

        Dense(units=256, activation='relu'),
        Dense(units=64, activation='relu'),
        Dropout(0.25),
        Dense(units=6, activation='softmax')
    ])

    return model


CreateModel = create_model()

print('Model Details are : ')
print(CreateModel.summary())

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), patience=5)
optimizer = Adam(learning_rate=0.001)

CreateModel.compile(optimizer=optimizer, loss=CategoricalCrossentropy(), metrics=['accuracy'])
modelCheckpoint = ModelCheckpoint(filepath="models/model_{epoch:02d}-{val_loss:.2f}-{val_accuracy:.2f}.h5",
                                  verbose=0, save_best_only=False)
callback = [reduce_lr, modelCheckpoint]

history = CreateModel.fit(train_generator, epochs=50, validation_data=validation_generator,
                         verbose=1,
                         callbacks=callback)


train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

train_loss = history.history['loss']
val_loss = history.history['val_loss']

learning_rate = history.history['lr']

fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(12, 10))


ax[0].set_title('Training Accuracy as Epochs')
ax[0].plot(train_accuracy, 'o-', label='Train Accuracy')
ax[0].plot(val_accuracy, 'o-', label='Validation Accuracy')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Accuracy')
ax[0].legend(loc='best')

ax[1].set_title('Training/Validation Loss as Epochs')
ax[1].plot(train_loss, 'o-', label='Train Loss')
ax[1].plot(val_loss, 'o-', label='Validation Loss')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Loss')
ax[1].legend(loc='best')

ax[2].set_title('Learning Rate vs. Epochs')
ax[2].plot(learning_rate, 'o-', label='Learning Rate')
ax[2].set_xlabel('Epochs')
ax[2].set_ylabel('Loss')
ax[2].legend(loc='best')

plt.tight_layout()
plt.savefig(f'Z:\Desktop\学习笔记\CV\深度学习/photos/all_data.jpg')
plt.show()


def dis_acc():
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['accuracy', 'val_accuracy'], loc='upper left')
    plt.savefig(f'Z:\Desktop\学习笔记\CV\深度学习/photos/acc_data.jpg')
    plt.show()


def dis_loss():
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(f'Z:\Desktop\学习笔记\CV\深度学习/photos/loss_data.jpg')
    plt.show()


dis_acc()
dis_loss()

predictions = CreateModel.predict(validation_generator)

test_loss, test_accuracy = CreateModel.evaluate(validation_generator, batch_size=32)

print(f"Test Loss:     {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

y_pred = np.argmax(predictions, axis=1)
y_true = validation_generator.classes

cf_mtx = confusion_matrix(y_true, y_pred)

group_counts = ["{0:0.0f}".format(value) for value in cf_mtx.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in cf_mtx.flatten() / np.sum(cf_mtx)]
box_labels = [f"{v1}\n({v2})" for v1, v2 in zip(group_counts, group_percentages)]
box_labels = np.asarray(box_labels).reshape(6, 6)

plt.figure(figsize=(12, 10))
sns.heatmap(cf_mtx, xticklabels=labels.values(), yticklabels=labels.values(),
            cmap="YlGnBu", fmt="", annot=box_labels)
plt.xlabel('Predicted Classes')
plt.ylabel('True Classes')
plt.savefig(f'Z:\Desktop\学习笔记\CV\深度学习/photos/hxjz_data.jpg')
plt.show()

print(classification_report(y_true, y_pred, target_names=labels.values()))
