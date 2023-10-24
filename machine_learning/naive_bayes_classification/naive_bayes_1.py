"""
高斯分布 朴素贝叶斯
朴素贝叶斯分类是一种依据统计概率理论而实现的一种分类方式。
"""
import numpy as np
import matplotlib.pyplot as mp
import sklearn.naive_bayes as nb
import sklearn.model_selection as ms

# 整理样本
data = np.loadtxt(
	'dataset\multiple1.txt', delimiter=',')
x = data[:, :2].astype('f8')
y = data[:, -1].astype('f8')        #“f8”表示float64的数据类型
print(x.shape, y.shape)

#划分训练集、测试集
train_x, test_x, train_y, test_y = \
              ms.train_test_split( x,y,test_size=0.25, stratify=y, random_state=7)


# 创建高斯分布朴素贝叶斯分类器
model = nb.GaussianNB()


#交叉验证，看一下预测评估指标
ac_score = ms.cross_val_score(model, train_x, train_y, cv=5, scoring='accuracy')  # cv = 折叠数， 表示交叉验证次数
print("精确度：", ac_score.mean())
ac_score = ms.cross_val_score(model, train_x, train_y, cv=5, scoring='precision_weighted')  
print("查准率：", ac_score.mean())
ac_score = ms.cross_val_score(model, train_x, train_y, cv=5, scoring='recall_weighted')  
print("召回率：", ac_score.mean())
ac_score = ms.cross_val_score(model, train_x, train_y, cv=5, scoring='f1_weighted')  
print("f1得分：", ac_score.mean())

#用训练集训练模型
model.fit(train_x, train_y)

#输出模型的预测效果
pred_test_y = model.predict(test_x)
acc = (test_y == pred_test_y).sum() / test_y.size
print("准确度：", acc)

# 画图
mp.figure('Naive Bayes Classification', facecolor='lightgray')
mp.title('Naive Bayes Classification', fontsize=16)
# 绘制分类边界线
#把可视区间划分为500*500，
n = 500
l, r = x[:,0].min()-1, x[:,0].max()+1
b, t = x[:,1].min()-1, x[:,1].max()+1
grid_x, grid_y = np.meshgrid(np.linspace(l, r, n), np.linspace(b, t, n))

# 根据业务，模拟预测
#使用模型，得到点阵中每个坐标的类别
mesh_x = np.column_stack((grid_x.ravel(), grid_y.ravel()))   #(250000,2)

grid_z = model.predict(mesh_x)                               #(250000,)

grid_z = grid_z.reshape(grid_x.shape)                        # 把grid_z 变维：(500,500)

mp.pcolormesh(grid_x, grid_y, grid_z, cmap='gray')
mp.scatter(test_x[:,0], test_x[:,1], s=80, c=test_y, cmap='brg_r', label='Samples')
mp.legend()
mp.show()