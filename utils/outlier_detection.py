import numpy as np
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerPathCollection

def update_legend_marker_size(handle, orig):
    "Customize size of the legend marker"
    handle.update_from(orig)
    handle.set_sizes([20])

# 数据异常值检测
def main():
    # 1. 数据构造
    np.random.seed(42)
    # 构造正常值数据
    X_inliers = 0.3 * np.random.randn(100, 2)
    X_inliers = np.r_[X_inliers + 2, X_inliers - 2]
    # 构造异常值数据
    X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
    # 拼接数据
    X = np.r_[X_inliers, X_outliers]

    # 为数据定义标签(1: 正常值，-1: 异常值)
    n_outliers = len(X_outliers)
    ground_truth = np.ones(len(X), dtype=int)
    ground_truth[-n_outliers:] = -1

    
    # 2. 建立异常值检测模型
    clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    # 预测数据中哪些为异常值(1: 正常值， -1: 异常值)
    y_pred = clf.fit_predict(X)
    # 查看异常值数据的索引和个数
    outlier_indices = np.where(y_pred == -1)
    num_pred_outlier = outlier_indices[0].shape
    print("根据模型预测，总体数据中异常值数据的个数为{}, 其数据标号为{}".format(num_pred_outlier, outlier_indices))
    # 查看误差
    n_errors = (y_pred != ground_truth).sum()
    # 查看每个数据点为异常值的分数
    X_scores = clf.negative_outlier_factor_
    
    # 3. 对ground_truth和pred的数据进行可视化
    # 3.1 将正常值标记为红色，异常值标记为蓝色(gt)
    plt.scatter(X[:, 0], X[:, 1], c=ground_truth, cmap=plt.cm.coolwarm, s=10, label="Data points")
    # plot circles with radius proportional to the outlier scores
    radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
    # 将数据预测的异常值分数转为半径，绘制在数据点上(半径越大，该数据点为异常值的概率越大)
    scatter = plt.scatter(
        X[:, 0],
        X[:, 1],
        s=1000 * radius,
        edgecolors="r",
        facecolors="none",
        label="Outlier scores",
    )
    plt.axis("tight")
    plt.xlim((-5, 5))
    plt.ylim((-5, 5))
    plt.xlabel("prediction errors: %d" % (n_errors))
    plt.legend(
        handler_map={scatter: HandlerPathCollection(update_func=update_legend_marker_size)}
    )
    plt.title("Local Outlier Factor (LOF)")
    plt.savefig('./outlier_detection.png')
    
    plt.clf() # 清空图像
    
    # 3.2 将正常值标记为红色，异常值标记为蓝色(y_pred)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap=plt.cm.coolwarm, s=10, label="Data points")
    # plot circles with radius proportional to the outlier scores
    radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
    # 将数据预测的异常值分数转为半径，绘制在数据点上(半径越大，该数据点为异常值的概率越大)
    scatter = plt.scatter(
        X[:, 0],
        X[:, 1],
        s=1000 * radius,
        edgecolors="r",
        facecolors="none",
        label="Outlier scores",
    )
    plt.axis("tight")
    plt.xlim((-5, 5))
    plt.ylim((-5, 5))
    plt.xlabel("prediction errors: %d" % (n_errors))
    plt.legend(
        handler_map={scatter: HandlerPathCollection(update_func=update_legend_marker_size)}
    )
    plt.title("Local Outlier Factor (LOF)")
    plt.savefig('./outlier_detection_pred.png')
    
    
    
if __name__ == "__main__":
    main()