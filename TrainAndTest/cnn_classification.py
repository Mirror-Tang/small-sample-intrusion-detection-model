import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input
from tensorflow.keras.utils import to_categorical
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 加载数据
train_data = pd.read_csv('normalized_train_data.csv')
test_data = pd.read_csv('normalized_test_data.csv')

# 假设最后一列是标签列，将其分离出来
train_labels = train_data.iloc[:, -1].values
train_data = train_data.iloc[:, :-1].values

test_labels = test_data.iloc[:, -1].values
test_data = test_data.iloc[:, :-1].values

# 将标签转换为分类格式
num_classes = len(np.unique(train_labels))
train_labels = to_categorical(train_labels, num_classes=num_classes)
test_labels = to_categorical(test_labels, num_classes=num_classes)

# 调整数据形状以适应1D卷积层
train_data = np.expand_dims(train_data, axis=2)
test_data = np.expand_dims(test_data, axis=2)

# 创建CNN模型
input_shape = train_data.shape[1:]
input_layer = Input(shape=input_shape)

x = Conv1D(64, 3, activation='relu')(input_layer)
x = MaxPooling1D(pool_size=2)(x)

x = Conv1D(128, 3, activation='relu')(x)
x = MaxPooling1D(pool_size=2)(x)

x = Conv1D(256, 3, activation='relu')(x)
x = MaxPooling1D(pool_size=2)(x)

x = Flatten()(x)

x = Dense(100, activation='relu')(x)
x = Dense(20, activation='relu')(x)
x = Dense(5, activation='relu')(x)
output_layer = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=input_layer, outputs=output_layer)

# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(train_data, train_labels, epochs=10, batch_size=32, validation_data=(test_data, test_labels))

# 提取中间特征
intermediate_layer_model = Model(inputs=model.input, outputs=model.layers[-4].output)
intermediate_features_train = intermediate_layer_model.predict(train_data)
intermediate_features_test = intermediate_layer_model.predict(test_data)

# 创建并训练SVM和k-NN分类器
svm = SVC()
knn = KNeighborsClassifier(n_neighbors=1)

svm.fit(intermediate_features_train, np.argmax(train_labels, axis=1))
knn.fit(intermediate_features_train, np.argmax(train_labels, axis=1))

# 评估分类器
svm_predictions = svm.predict(intermediate_features_test)
knn_predictions = knn.predict(intermediate_features_test)

svm_accuracy = accuracy_score(np.argmax(test_labels, axis=1), svm_predictions)
knn_accuracy = accuracy_score(np.argmax(test_labels, axis=1), knn_predictions)

print(f"SVM 分类准确度: {svm_accuracy}")
print(f"k-NN 分类准确度: {knn_accuracy}")