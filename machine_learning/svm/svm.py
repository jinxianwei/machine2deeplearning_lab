import os
import sys
current_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, current_path)
import matplotlib.pyplot as plt

from utils.read_csv_data import read_csv_data
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn import svm


def main():
    # 1. 读取csv数据
    name_dict, data = read_csv_data("dataset/iris.csv")
    # 鸢尾花数据集三类标签
    label_dict = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    label_list = ['setosa', 'versicolor', 'virginica']
    
    # 2. 确定特征和标签
    x = data[:, :2] # 只选择前两个特征进行训练，为方便可视化
    y = data[:, -1]
    
    # 3. 划分训练集和测试集
    x_train, x_test, y_train, y_test= train_test_split(x, y, stratify=y, test_size=0.5, random_state=0)
    
    # 4. 创建模型
    # we create an instance of SVM and fit out data. We do not scale our
    # data since we want to plot the support vectors
    C = 1.0  # SVM regularization parameter
    models = (
        svm.SVC(kernel="linear", C=C),
        svm.LinearSVC(C=C, max_iter=10000, dual="auto"),
        svm.SVC(kernel="rbf", gamma=0.7, C=C),
        svm.SVC(kernel="poly", degree=3, gamma="auto", C=C),
    )
    # 5. 模型训练
    models = (clf.fit(x_train, y_train) for clf in models)

    # 6. 绘图
    # title for the plots
    titles = (
        "SVC with linear kernel",
        "LinearSVC (linear kernel)",
        "SVC with RBF kernel",
        "SVC with polynomial (degree 3) kernel",
    )

    # Set-up 2x2 grid for plotting.
    fig, sub = plt.subplots(2, 2)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    # 绘制x_test预测边界，并绘制x_test真实标签y_test的散点图
    for clf, title, ax in zip(models, titles, sub.flatten()):
        # 预测评估指标
        y_pred = clf.predict(x_test)
        accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
        precision = precision_score(y_true=y_test, y_pred=y_pred, average='macro')
        recall = recall_score(y_true=y_test, y_pred=y_pred, average='macro')
        f1 = f1_score(y_true=y_test, y_pred=y_pred, average='macro')
        print(f"模型[{title}] 在测试集上的精确率为{accuracy}, 准确度为{precision}, 召回率为{recall}, F1分数为{f1}")
        
        # 绘图
        disp = DecisionBoundaryDisplay.from_estimator(
            clf,
            x_test,
            response_method="predict",
            cmap=plt.cm.coolwarm,
            alpha=0.8,
            ax=ax,
            xlabel=label_list[0],
            ylabel=label_list[1],
        )
        ax.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)

    plt.savefig("./svm.png")

if __name__ == "__main__":
    main()