import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from read_csv_data import read_csv_data
from sklearn.model_selection import GridSearchCV
    
def main():
    # 1. 读取csv数据
    name_dict, data = read_csv_data("dataset/iris.csv")    
    # 2. 建立异常值检测模型, 其中模型的参数需要适当调整
    clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    # 预测数据中哪些为异常值(1: 正常值， -1: 异常值)
    y_pred = clf.fit_predict(data)
    # 查看异常值数据的索引和个数
    outlier_indices = np.where(y_pred == -1)
    num_pred_outlier = outlier_indices[0].shape
    print("根据模型预测，总体数据中异常值数据的个数为{}, 其数据标号为{}".format(num_pred_outlier, outlier_indices))
    
    # TODO 3. 找一个比较合适的模型参数
    # 定义参数范围
    param_grid = {'n_neighbors': [5, 10, 15, 20],
                'contamination': [0.05, 0.1, 0.15, 0.2]}
    # 创建 LOF 模型
    lof = LocalOutlierFactor()
    # 使用 GridSearchCV 寻找最佳参数
    grid_search = GridSearchCV(lof, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)
    grid_search.fit(data)

    # 输出最佳参数
    print("Best parameters:", grid_search.best_params_)
        
    
    
    
    
    
if __name__ == "__main__":
    main()