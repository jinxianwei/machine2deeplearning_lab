import os
import sys
current_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, current_path)

import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from utils.read_csv_data import read_csv_data
from sklearn.model_selection import train_test_split

def main():
    # 1. 读取csv数据
    name_dict, data = read_csv_data("dataset/iris.csv")
    # 鸢尾花数据集三类标签
    label_dict = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    label_list = ['setosa', 'versicolor', 'virginica']
    feature_name = [val for key, val in name_dict.items()]
    
    # 2. 确定特征和标签
    x = data[:, :2] # 只选择前两个特征进行训练，为方便可视化
    y = data[:, -1]
    
    # 3. 划分训练集和测试集
    x_train, x_test, y_train, y_test= train_test_split(x, y, stratify=y, test_size=0.5, random_state=0)

    n_features = x.shape[1]

    # 4. 创建模型
    C = 10
    kernel = 1.0 * RBF([1.0, 1.0])  # for GPC
    # 创建不同类别的分类器
    classifiers = {
        "L1 logistic": LogisticRegression(
            C=C, penalty="l1", solver="saga", multi_class="multinomial", max_iter=10000
        ),
        "L2 logistic (Multinomial)": LogisticRegression(
            C=C, penalty="l2", solver="saga", multi_class="multinomial", max_iter=10000
        ),
        "L2 logistic (OvR)": LogisticRegression(
            C=C, penalty="l2", solver="saga", multi_class="ovr", max_iter=10000
        ),
        "Linear SVC": SVC(kernel="linear", C=C, probability=True, random_state=0),
        "GPC": GaussianProcessClassifier(kernel),
    }

    n_classifiers = len(classifiers)

    plt.figure(figsize=(3 * 2, n_classifiers * 2))
    plt.subplots_adjust(bottom=0.2, top=0.95)

    xx = np.linspace(3, 9, 100) # x[:, 0].min() ~ x[:, 0].max()
    yy = np.linspace(1, 5, 100).T # x[:, 1].min() ~ x[:, 1].max() 为了可视化样本空间，需要考虑特征的最大最小数值区间
    xx, yy = np.meshgrid(xx, yy)
    Xfull = np.c_[xx.ravel(), yy.ravel()] # 正交得到样本空间的待预测点

    # 5. 对不同模型进行训练
    for index, (name, classifier) in enumerate(classifiers.items()):
        # 训练模型
        classifier.fit(x_train, y_train)
        # 在测试集上进行预测
        y_pred = classifier.predict(x_test)
        accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
        print("Accuracy (test) for %s: %0.1f%% " % (name, accuracy * 100))

        # View probabilities:
        # 对离散空间的所有样本点进行预测
        probas = classifier.predict_proba(Xfull)
        n_classes = np.unique(y_pred).size
        for k in range(n_classes):
            plt.subplot(n_classifiers, n_classes, index * n_classes + k + 1)
            plt.title("{}".format(label_dict[k]))
            if k == 0:
                plt.ylabel(name)
            imshow_handle = plt.imshow(
                probas[:, k].reshape((100, 100)), extent=(3, 9, 1, 5), origin="lower"
            )
            plt.xticks(())
            plt.yticks(())
            idx = y_pred == k # 得到预测类别为k的样本索引True，忽略其他预测类别的样本(检查x_test[idx, 0].shape)
            if idx.any():
                # 绘制预测类别为k的测试集样本散点
                plt.scatter(x_test[idx, 0], x_test[idx, 1], marker="o", c="w", edgecolor="k")
            if k == 2:
                plt.xlabel(feature_name[0])
                plt.ylabel(feature_name[1])

    ax = plt.axes([0.15, 0.04, 0.7, 0.05])
    plt.title("Probability")
    plt.colorbar(imshow_handle, cax=ax, orientation="horizontal")

    plt.savefig('./prob.png')

if __name__ == "__main__":
    main()