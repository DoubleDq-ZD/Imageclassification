import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# sns.set(style="whitegrid")

import tensorflow as tf
import tensorflow.keras

from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from sklearn.metrics import confusion_matrix, classification_report
from random import randint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

validation_datagen = ImageDataGenerator(rescale=1.0 / 255)
validation_generator = validation_datagen.flow_from_directory('seg_data/seg_test/seg_test/',
                                                              target_size=(224, 224),
                                                              batch_size=50,
                                                              shuffle=True)
labels = {value: key for key, value in validation_generator.class_indices.items()}
CreateModel = tf.keras.models.load_model('models/model_50-0.33-0.90.h5')
predictions = CreateModel.predict(validation_generator)
test_loss, test_accuracy = CreateModel.evaluate(validation_generator, batch_size=32)

print(f"Test Loss:     {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

y_pred = np.argmax(predictions, axis=1)
y_true = validation_generator.classes
print(y_pred)
print(y_true)
#
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
plt.show()

print(classification_report(y_true, y_pred, target_names=labels.values()))
