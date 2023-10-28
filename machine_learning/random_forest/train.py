"""
混凝土 随机森林
"""
import numpy as np
import matplotlib.pyplot as mp
import sklearn.ensemble as se
import sklearn.metrics as sm
import sklearn.utils as su

data = []
with open('../machine2deeplearning_lab/dataset/npvproject-concrete.csv', 'r') as f:
	for line in f.readlines():
		data.append(line[:-1].split(','))
# 通过data整理输入、输出、header
data = np.array(data)
header = data[0, 0:8]
x = data[1:, 0:8].astype('f8')
y = data[1:, -1].astype('f8')

# 整理训练集、测试集  训练随机森林回归模型
x, y = su.shuffle(x, y, random_state=7)
train_size = int(len(x) * 0.75)
train_x, test_x, train_y, test_y = \
	x[:train_size], x[train_size:], \
	y[:train_size], y[train_size:]

model = se.RandomForestRegressor(
	max_depth=10, n_estimators=1000,
	min_samples_split=2)
model.fit(train_x, train_y)
pred_test_y = model.predict(test_x)
sample =[[540.0,0.0,0.0,162.0,2.5,1040.0,676.0,28]]
z = model.predict(sample)
print (z)
# 评估模型
print("r2得分{}",sm.r2_score(test_y, pred_test_y))

#特征重要性
concrete_fi = model.feature_importances_

mp.figure('Feature Importance', facecolor='lightgray')
mp.subplot()
mp.title('concrete Feature Importance', fontsize=16)
mp.grid(linestyle=':', axis='y')
x = np.arange(concrete_fi.size)
sorted_indices = concrete_fi.argsort()[::-1]
mp.xticks(x, header[sorted_indices])
mp.bar(x, concrete_fi[sorted_indices], 0.8, 
	color='orangered', label='concrete Feature Importances')
mp.legend()
mp.tight_layout()
mp.show()