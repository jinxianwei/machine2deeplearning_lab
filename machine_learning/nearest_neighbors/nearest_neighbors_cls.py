from sklearn.model_selection import train_test_split

import os
import sys
current_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, current_path)

from utils.read_csv_data import read_csv_data
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay


def main():
    # 1. 读取csv数据
    name_dict, data = read_csv_data("dataset/iris.csv")
    print(name_dict, data.shape)
    # 鸢尾花数据集三类标签
    label_dict = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    label_list = ['setosa', 'versicolor', 'virginica']
    
    # 2. 确定特征和标签
    x = data[:, :2] # 只选择前两个特征进行训练，为方便可视化
    y = data[:, -1]
    
    # 3. 划分训练集和测试集
    x_train, x_test, y_train, y_test= train_test_split(x, y, stratify=y, test_size=0.5, random_state=0)
    
    # 4. 对数据进行归一化，并构建k近邻分类器
    clf = Pipeline(steps=[("scaler", StandardScaler()), 
                          ("knn", KNeighborsClassifier(n_neighbors=11))])
    
    # 4. 训练两种模型，并可视化分类边界
    _, axs = plt.subplots(ncols=2, figsize=(12, 5))
    for ax, weights in zip(axs, ("uniform", "distance")):
        # 训练模型
        clf.set_params(knn__weights=weights).fit(x_train, y_train)
        # 绘制在测试集上决策边界预测的效果
        disp = DecisionBoundaryDisplay.from_estimator(
            clf,
            x_test,
            response_method="predict",
            plot_method="pcolormesh",
            xlabel=name_dict[0],
            ylabel=name_dict[1],
            shading="auto",
            alpha=0.5,
            ax=ax,
        )
        # 按类别绘制测试集真实散点图
        scatter = disp.ax_.scatter(x_test[:, 0], x_test[:, 1], c=y_test, edgecolors="k")
        # 绘制左下角图框
        disp.ax_.legend(
            scatter.legend_elements()[0],
            label_list,
            loc="lower left",
            title="Classes",
        )
        # 设置子图标题
        _ = disp.ax_.set_title(
            f"3-Class classification\n(k={clf[-1].n_neighbors}, weights={weights!r})"
        )
    plt.savefig('./k_neighbors.png')
    
if __name__ == "__main__":
    main()
    