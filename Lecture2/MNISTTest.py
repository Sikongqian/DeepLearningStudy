# 加载 Keras 中的 MNIST 数据集
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(train_images.ndim)
print(train_images.shape)
print(train_images.dtype)
digit = train_images[4];
import matplotlib.pyplot as plt #根据像素点绘制图像（好有意思）
import numpy as np
plt.imshow(digit, plt.cm.binary)
plt.show();

#numpy 中操作张量
#切片运算
my_slice = train_images[10 : 100]
print(my_slice.shape)#(90,28,28)
#图片右下角14*14
my_slice = train_images[:, 14:, 14:]

#矩阵相乘的程序
def native_matrix_dot (x,y):
    assert len(x.shape) == 2
    assert len(y.shape) == 2
    assert x.shape[1] == y.shape[0]

    z = np.zeros((x.shape[0],y.shape[1]))
    for i in range(x.shpe[0]):
        for j in range(y.shape[1]):
            row_x = x[i,:]
            column_y = y[:,j]
            z[i,j] = native_vector_dot(row_x, column_y)
    return z
