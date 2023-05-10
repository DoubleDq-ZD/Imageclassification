import tensorflow as tf
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model


def loadImage(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_AREA)
    img = img.astype("float32")
    img /= 255.
    return np.array(img[:, :, :3])


if __name__ == "__main__":
    LABELS = os.listdir('seg_data/seg_test/seg_test')
    model = load_model("models/model_50-0.33-0.90.h5")
    model.summary()
    path = 'seg_data/seg_test/seg_test/sea/20072.jpg'
    img = loadImage(path)
    res = np.argmax(model.predict(np.array([img])))
    print(f"预测结果：{LABELS[res]}")