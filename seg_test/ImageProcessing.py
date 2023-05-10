import os
import shutil
import random
import glob
import numpy as np
np.seterr(divide='ignore',invalid='ignore')

import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
    fill_mode='nearest',  # 默认值，操作导致图像缺失时填充方式。constant,nearest,eflect,wrap
    channel_shift_range=0.0,  # 浮点数[0.0,255.0]，图像上色
    rotation_range=60,  # 指定旋转角度范围
    zca_whitening=False,  # 是否应用 ZCA 白化。
    brightness_range=[0.3, 1.5],  # 随机调整亮度
    width_shift_range=0.1,  # 水平平移百分比，不宜太大一般0.1,0.2
    height_shift_range=0.1,  # 垂直平移百分比，不宜太大一般0.1,0.2
    zoom_range=[0.9, 1],  # 随机缩放范围
    horizontal_flip=True,  # 随机对图片执行水平翻转操作
    vertical_flip=False,  # 对图片执行上下翻转操作
    shear_range=15,  # 错切变换角度。
    rescale=1. / 255,  # 缩放
    data_format='channels_last')

def gauss(img_path):
    x = cv2.imread(img_path)
    y = cv2.resize(x, (256, 256))
    n = np.random.normal(0, 30, size=y.shape)

    y = y.astype("float")
    y = y + n
    y = np.where(y > 255, 255, y)
    y = np.where(y < 0, 0, y)
    y = y.astype('uint8')
    cv2.imwrite(f'{img_path[:-4]}_gs.jpg', y)


# BGR -> HSV
def BGR2HSV(_img):
    img = _img.copy() / 255.

    hsv = np.zeros_like(img, dtype=np.float32)

    # get max and min
    max_v = np.max(img, axis=2).copy()
    min_v = np.min(img, axis=2).copy()
    min_arg = np.argmin(img, axis=2)

    # H
    hsv[..., 0][np.where(max_v == min_v)] = 0
    ## if min == B
    ind = np.where(min_arg == 0)
    hsv[..., 0][ind] = 60 * (img[..., 1][ind] - img[..., 2][ind]) / (max_v[ind] - min_v[ind]) + 60
    ## if min == R
    ind = np.where(min_arg == 2)
    hsv[..., 0][ind] = 60 * (img[..., 0][ind] - img[..., 1][ind]) / (max_v[ind] - min_v[ind]) + 180
    ## if min == G
    ind = np.where(min_arg == 1)
    hsv[..., 0][ind] = 60 * (img[..., 2][ind] - img[..., 0][ind]) / (max_v[ind] - min_v[ind]) + 300

    # S
    hsv[..., 1] = max_v.copy() - min_v.copy()

    # V
    hsv[..., 2] = max_v.copy()

    return hsv


def HSV2BGR(_img, hsv):
    img = _img.copy() / 255.

    # get max and min
    max_v = np.max(img, axis=2).copy()
    min_v = np.min(img, axis=2).copy()

    out = np.zeros_like(img)

    H = hsv[..., 0]
    S = hsv[..., 1]
    V = hsv[..., 2]

    C = S
    H_ = H / 60.
    X = C * (1 - np.abs(H_ % 2 - 1))
    Z = np.zeros_like(H)

    vals = [[Z, X, C], [Z, C, X], [X, C, Z], [C, X, Z], [C, Z, X], [X, Z, C]]

    for i in range(6):
        ind = np.where((i <= H_) & (H_ < (i + 1)))
        out[..., 0][ind] = (V - C)[ind] + vals[i][0][ind]
        out[..., 1][ind] = (V - C)[ind] + vals[i][1][ind]
        out[..., 2][ind] = (V - C)[ind] + vals[i][2][ind]

    out[np.where(max_v == min_v)] = 0
    out = np.clip(out, 0, 1)
    out = (out * 255).astype(np.uint8)

    return out

def Hsv_p(img_path):
    img = cv2.imread(img_path).astype(np.float32)
    hsv = BGR2HSV(img)
    hsv[..., 0] = (hsv[..., 0] + 180) % 360
    out = HSV2BGR(img, hsv)
    cv2.imwrite(f"{img_path[:-4]}_hsv.jpg", out)

def NEW_HSV(img_path):
    img = cv2.imread(img_path)
    IMG = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.imwrite(f'{img_path[:-4]}_newHsv.jpg',IMG)

def add_data(path):
    subdirs = os.listdir(path)
    subdirs.sort()
    print(subdirs)
    for index in range(len(subdirs)):
        subdir = os.path.join(path, subdirs[index])
        # for img_path in glob.glob("{}/*.jpg".format(subdir)):
        #     gen_data(subdir, img_path)
        for img_path in glob.glob("{}/*.jpg".format(subdir)):
            gauss(img_path)
            NEW_HSV(img_path)
        for img_path in glob.glob("{}/*gs*.jpg".format(subdir)):
            Hsv_p(img_path)
        print(subdir,'文件夹扩充完毕！')
    print('Done!')


def gen_data(subdir, path):
    # 载入图片
    img = load_img(path)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    i = 0
    for batch in datagen.flow(x, batch_size=1, save_to_dir=subdir, save_format='jpg'):
        i += 1
        if i >= 3:
            break

