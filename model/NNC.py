import pandas as pd
import numpy as np
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

# 创建并训练SVM分类器
svm = SVC()
svm.fit(train_data, train_labels)

# 创建并训练k-NN分类器
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(train_data, train_labels)

# 评估SVM分类器
svm_predictions = svm.predict(test_data)
svm_accuracy = accuracy_score(test_labels, svm_predictions)
print(f"SVM 分类准确度: {svm_accuracy}")

# 评估k-NN分类器
knn_predictions = knn.predict(test_data)
knn_accuracy = accuracy_score(test_labels, knn_predictions)
print(f"k-NN 分类准确度: {knn_accuracy}")