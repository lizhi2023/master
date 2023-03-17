data = pd.read_excel(r'./训练top20.xlsx')
data2 = pd.read_excel(r'./验证top20.xlsx')

ss = StandardScaler()  # 标准化

feature = ss.fit_transform(data.iloc[:, 1:])  # 训练集属性
labels = data.iloc[:, 0]  # 训练集标签
feature2 = ss.transform(data2)  # 预测集属性

# 20%用作测试集，其余的用作训练集
X_train, X_test, Y_train, Y_test = train_test_split(feature, labels, test_size=0.20)
# 使用RandomOverSampler从少数类的样本中进行随机采样来增加新的样本使各个分类均衡
ros = RandomOverSampler()
X_resampled, y_resampled = ros.fit_resample(X_train, Y_train)

reg_before = XGBC()
reg_before.fit(X_resampled, y_resampled)

print("测试集和训练集的分类精度分别为：", reg_before.score(X_test, Y_test), reg_before.score(X_resampled, y_resampled))
print("test_f1:", metrics.f1_score(reg_before.predict(X_test), Y_test))
print("train_f1", metrics.f1_score(reg_before.predict(X_resampled), y_resampled))

