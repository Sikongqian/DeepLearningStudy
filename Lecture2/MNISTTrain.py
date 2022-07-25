# 加载 Keras 中的 MNIST 数据集
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(train_images.ndim)
print(train_images.shape)
print(train_images.dtype)
# 网络架构
from keras import models
from keras import layers
network = models.Sequential()
# 两个dense层，对输入数据进行一些简单的张量运算
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

# 编译步骤
# categorical——crossenropy 损失函数 rmsprop梯度下降优化器
network.compile(optimizer='rmsprop',
loss='categorical_crossentropy',
metrics=['accuracy'])

# 准备图像数据
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

# 准备标签(现在keras完全置于tf模块中，要从根模块引入)
from tensorflow.keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 训练
# 迭代五次，每次128大小
network.fit(train_images, train_labels, epochs=5, batch_size=128)

# 检测
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)