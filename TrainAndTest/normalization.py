import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# 加载数据
data = pd.read_csv('kdd_nsl_data.csv')

# 分离出离散值和连续值特征
discrete_features = ['protocol_type', 'service', 'flag']
continuous_features = [col for col in data.columns if col not in discrete_features + ['split']]

# 对离散特征进行独热编码
one_hot_encoder = OneHotEncoder(sparse=False)
encoded_discrete_features = one_hot_encoder.fit_transform(data[discrete_features])

# 将编码后的特征转换为DataFrame并添加列名
encoded_discrete_features_df = pd.DataFrame(encoded_discrete_features, columns=one_hot_encoder.get_feature_names(discrete_features))

# 对连续特征进行标准化
scaler = StandardScaler()

# 只使用训练数据来计算均值和标准差
train_data = data[data['split'] == 'train']
test_data = data[data['split'] == 'test']

# 拟合训练数据并转换
train_continuous_features = scaler.fit_transform(train_data[continuous_features])

# 使用训练数据的均值和标准差转换测试数据
test_continuous_features = scaler.transform(test_data[continuous_features])

# 将标准化后的特征转换为DataFrame并添加列名
train_continuous_features_df = pd.DataFrame(train_continuous_features, columns=continuous_features)
test_continuous_features_df = pd.DataFrame(test_continuous_features, columns=continuous_features)

# 合并处理后的特征
processed_train_data = pd.concat([train_continuous_features_df.reset_index(drop=True), 
                                  encoded_discrete_features_df.iloc[train_data.index].reset_index(drop=True)], axis=1)
processed_test_data = pd.concat([test_continuous_features_df.reset_index(drop=True), 
                                 encoded_discrete_features_df.iloc[test_data.index].reset_index(drop=True)], axis=1)

# 保存处理后的数据
processed_train_data.to_csv('normalized_train_data.csv', index=False)
processed_test_data.to_csv('normalized_test_data.csv', index=False)

print("数据处理完成，已保存为normalized_train_data.csv和normalized_test_data.csv")