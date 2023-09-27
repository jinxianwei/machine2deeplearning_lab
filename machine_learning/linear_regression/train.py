import os
import sys
current_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, current_path)

from utils.read_csv_data import read_csv_data
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def main():
    # 1. 读取csv数据
    name_dict, data = read_csv_data("dataset/npvproject-concrete.csv")
    print(name_dict, data.shape)
    
    # 2. 确定特征和标签
    X = data[:, 0: -1]
    y = data[:, -1]
    
    # 3. 划分训练集和测试集
    # TODO 数据集是否划分取决于模型，线性回归模型没有划分数据集的必要
    x_train, x_test, y_train, y_test= train_test_split(X, y, test_size=0.5, random_state=0)
    
    # 4. 建立模型训练
    model = LinearRegression()
    model.fit(x_train, y_train)
    
    # 5. 得到结果
    print("系数 {}".format(model.coef_))
    print("截距 {}".format(model.intercept_))
    
    # 6. 计算损失
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_true=y_test, y_pred=y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
    r2 = r2_score(y_true=y_test, y_pred=y_pred)
    
    print(f"均方误差为{mse}, 均方根误差为{rmse}, 平方绝对误差为{mae}, R^2分数为{r2}")
    
    
    
if __name__ == "__main__":
    main()