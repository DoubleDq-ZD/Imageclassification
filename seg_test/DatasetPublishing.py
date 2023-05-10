import os
import matplotlib.pyplot as plt


def data_vit(path, kx):
    labels = os.listdir(path)
    nums = []
    for i in labels:
        nums.append(len(os.listdir(os.path.join(path, i))))
    print(labels, nums)

    fig = plt.figure()
    plt.bar(labels, nums, color='b')
    for a, b in zip(labels, nums):
        plt.text(a, b, "%.2f" % b, ha='center')
    plt.title('data')
    plt.savefig(f'Z:\Desktop\学习笔记\CV\深度学习/photos/{kx}_data.jpg')
    plt.show()


data_vit('seg_data/seg_test/seg_test', 'seg_test')
data_vit('seg_data/seg_train/seg_train', 'seg_train')
