import matplotlib.pyplot as plt
# 加载IMDB数据集
from keras.datasets import imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = 10000)


# 转换数据
import numpy as np
# 这里先创建一个对应大小的0矩阵，再填充
def vectorize_sequences (sequences, dimension = 10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):# 迭代器，i是第i个sequence，sequence是一个个的值
        results[i, sequence] = 1
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# 标签向量化，直接利用np中的asarray进行向量化
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# 网络构建 dense 层的堆叠
# 两个中间层， 每层16个隐藏单元， 最后一层输出标量
# 中间层使用relu作为激活函数。最后一层使用sigmoid激活输出
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16, activation = 'relu', input_shape = (10000,)))
model.add(layers.Dense(16, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))

# 编译模型 优化器，目标函数，指标
from keras import optimizers

model.compile(optimizer = 'rmsprop',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

# 流出验证集，检测训练时精度的变化
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# 训练模型
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs = 20,# 迭代20次
                    batch_size = 512,# 每批512
                    validation_data = (x_val, y_val))# 训练时验证集

# 画出训练时的曲线

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epoachs = range(1, len(loss_values) + 1)

plt.plot(epoachs, loss_values, 'bo'# 蓝色原点
         , label = 'Training loss')
plt.plot(epoachs, val_loss_values, 'b', label = 'Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend()
plt.show()






